import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.augmentations import functional as F
from albumentations import (
    HorizontalFlip, ShiftScaleRotate, CoarseDropout,
    GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, Cutout
)
import albumentations as A
from typing import List
import catalyst
from catalyst.dl import Callback, MetricCallback, CallbackOrder, CriterionCallback, RunnerState
import random
import collections
from catalyst.utils import set_global_seed
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, CriterionAggregatorCallback
from catalyst.contrib.optimizers import RAdam, Lookahead, Lamb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from catalyst.contrib.schedulers.onecycle import OneCycleLRWithWarmup
from efficientnet.model import EfficientNet
from efficientnet.utils import get_same_padding_conv2d, round_filters
from resnet.model import resnet50, resnext101_32x8d
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

set_global_seed(42)

DATA_FOLDER = './data'
TRAIN_DATA = './data/train'

train_data = pd.read_csv(os.path.join(DATA_FOLDER, './train_data.tsv'))
train_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
train_data.drop(['test_fold'], axis=1, inplace=True)
train_data['type'] = 'train'
validation_data = pd.read_csv(os.path.join(DATA_FOLDER, './validation_data.tsv'))
validation_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = validation_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
validation_data['type'] = 'validation'
all_data = pd.concat([train_data, validation_data])

train_image_ids = all_data['image_id'][all_data['type'] == 'train'].values
valid_image_ids = all_data['image_id'][all_data['type'] != 'train'].values
train_mask = all_data['type'] == 'train'
val_mask = all_data['type'] != 'train'

image_data = []
for i in range(4):
    chunk = pd.read_parquet(os.path.join(DATA_FOLDER, 'train_image_data_{}.parquet'.format(i)))
    chunk.index = chunk.image_id
    chunk.drop(['image_id'], axis=1, inplace=True)
    chunk.astype(np.uint8)
    image_data.append(chunk)
del chunk

image_data = pd.concat(image_data)

batch_size = 100
num_workers = 0

# augs = [MotionBlur(always_apply=True),
#         ShiftScaleRotate(always_apply=True),
#         GaussNoise(always_apply=True),
#         MedianBlur(always_apply=True),
#         CoarseDropout(always_apply=True),
#         # GridDropout(always_apply=True)
#         ]

# transforms_train = A.Compose([
#     AugMix(width=3,
#            depth=8,
#            alpha=.25,
#            p=1.,
#            augmentations=augs,
#            mean=[0.5],
#            std=[0.5],
#            resize_height=100,
#            resize_width=100),
# ])

transforms_train = A.Compose([
        A.Resize(width=128,
                height=128),
        A.OneOf([A.RandomContrast(), 
                 A.RandomBrightness(), 
                 A.RandomGamma(),
                 A.RandomBrightnessContrast()],p=0.5),
        A.OneOf([A.GridDistortion(),
                 A.ElasticTransform(), 
                 A.OpticalDistortion(),
                 A.ShiftScaleRotate(),
                ],p=0.5),
        A.CoarseDropout(),
        A.Normalize(mean=0.5,
                    std=0.5)
    ],p=1.0)

transforms_val = A.Compose([
    A.Resize(width=128,
             height=128),
    A.Normalize(mean=0.5,
                std=0.5)
])

train_dataset = ImageDataset(
    df = image_data.loc[train_image_ids,:],
    labels = all_data.loc[train_mask, :],
    label = 'all',
    transforms = transforms_train
    )
val_dataset = ImageDataset(
    df = image_data.loc[valid_image_ids,:],
    labels = all_data.loc[val_mask, :],
    label = 'all',
    transforms = transforms_val
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=False,
    shuffle=True
    )
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=False,
    shuffle=False
    )

model = EfficientNet.from_name("efficientnet-b8")
model.load_state_dict(torch.load('efficientnet/efficientnet-b8.pth'), strict=False)
Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
out_channels = round_filters(32, model._global_params)
model._conv_stem = Conv2d(1, out_channels, kernel_size=3, stride=2, bias=False)
model.cuda()

loaders = collections.OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = val_loader

runner = SupervisedRunner(
    input_key='image',
    output_key=None,
    input_target_key=None
    )

optimizer = RAdam(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.001
    )

scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=3)

criterions_dict = {
    'vowel_diacritic_loss':torch.nn.CrossEntropyLoss(), 
    'grapheme_root_loss':torch.nn.CrossEntropyLoss(),
    'consonant_diacritic_loss':torch.nn.CrossEntropyLoss()
    }
callbacks=[
    MixupCutmixCallback(fields=["image"], 
                        output_key=("logit_grapheme_root", "logit_vowel_diacritic", "logit_consonant_diacritic"),
                        input_key=("grapheme_root", "vowel_diacritic", "consonant_diacritic"),
                        mixuponly=False,
                        alpha=0.5),
    CriterionCallback(input_key='grapheme_root',
                    output_key='logit_grapheme_root',
                    prefix='grapheme_root_loss',
                    criterion_key='grapheme_root_loss', multiplier=2.0),
    CriterionCallback(input_key='vowel_diacritic',
                    output_key='logit_vowel_diacritic',
                    prefix='vowel_diacritic_loss',
                    criterion_key='vowel_diacritic_loss', 
                    multiplier=1.0),
    CriterionCallback(input_key='consonant_diacritic',
                    output_key='logit_consonant_diacritic',
                    prefix='consonant_diacritic_loss',
                    criterion_key='consonant_diacritic_loss', 
                    multiplier=1.0),
    CriterionAggregatorCallback(prefix='loss',
                                loss_keys=['grapheme_root_loss',
                                        'vowel_diacritic_loss',
                                        'consonant_diacritic_loss']),
    TaskMetricCallback(
        output_key=("logit_grapheme_root", "logit_vowel_diacritic", "logit_consonant_diacritic"),
        input_key=("grapheme_root", "vowel_diacritic", "consonant_diacritic"))]

runner.train(
    model=model,
    main_metric='loss',
    minimize_metric=True,
    criterion=criterions_dict,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders=loaders,
    logdir=os.path.join(DATA_FOLDER, './effnetb8_three_heavy_head_parquet_mixes'),
    scheduler=scheduler,
    fp16=True,
    num_epochs=200,
    verbose=True
    )

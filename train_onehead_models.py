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
from catalyst.dl.callbacks import EarlyStoppingCallback
from catalyst.contrib.optimizers import RAdam, Lookahead, Lamb
from efficientnet_pytorch.model import EfficientNet
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
    chunk = pd.read_feather(os.path.join(TRAIN_DATA, 'train_data_{}.feather'.format(i)))
    chunk.index = chunk.image_id
    chunk.drop(['image_id'], axis=1, inplace=True)
    chunk.astype(np.uint8)
    image_data.append(chunk)
del chunk

image_data = pd.concat(image_data)

batch_size = 512
num_workers = 4

augs = [HorizontalFlip(always_apply=True),
        MotionBlur(always_apply=True),
        ShiftScaleRotate(always_apply=True),
        GaussNoise(always_apply=True),
        MedianBlur(always_apply=True),
        CoarseDropout(always_apply=True),
        GridDropout(always_apply=True)]

transforms_train = A.Compose([
    AugMix(width=3,
           depth=8,
           alpha=.25,
           p=1.,
           augmentations=augs,
           mean=[0.5],
           std=[0.5],
           resize_height=100,
           resize_width=100),
])
transforms_val = A.Compose([
    A.Resize(width=100,
             height=100),
    A.Normalize(mean=0.5,
                std=0.5)
])

print("_____________Training grapheme root____________")

train_dataset = ImageDataset(
    df = image_data.loc[train_image_ids,:],
    labels = all_data.loc[train_mask, :],
    label = 'grapheme_root',
    transforms = transforms_train
    )
val_dataset = ImageDataset(
    df = image_data.loc[valid_image_ids,:],
    labels = all_data.loc[val_mask, :],
    label = 'grapheme_root',
    transforms = transforms_val
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=True
    )
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=False
    )

model = EfficientNet.from_pretrained("efficientnet-b3", num_classes=168, in_channels=1)
model.cuda()

loaders = collections.OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = val_loader

runner = SupervisedRunner()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
    )

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.1,
    patience=5
    )

criterion = torch.nn.CrossEntropyLoss()
callbacks=[
    RecallCallback(),
    EarlyStoppingCallback(patience = 7)
    ]

runner.train(
    model=model,
    main_metric='loss',
    minimize_metric=True,
    criterion=criterion,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders=loaders,
    logdir=os.path.join(DATA_FOLDER, './baseline_effnet_one_head/grapheme_root'),
    scheduler=scheduler,
    fp16=True,
    num_epochs=50,
    verbose=True
    )

del model

print("_____________Training vowel diacritic____________")

train_dataset = ImageDataset(
    df = image_data.loc[train_image_ids,:],
    labels = all_data.loc[train_mask, :],
    label = 'vowel_diacritic',
    transforms = transforms_train
    )
val_dataset = ImageDataset(
    df = image_data.loc[valid_image_ids,:],
    labels = all_data.loc[val_mask, :],
    label = 'vowel_diacritic',
    transforms = transforms_val
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=True
    )
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=False
    )

model = EfficientNet.from_pretrained("efficientnet-b3", num_classes=11, in_channels=1)
model.cuda()

loaders = collections.OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = val_loader

runner = SupervisedRunner()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
    )

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.1,
    patience=5
    )

criterion = torch.nn.CrossEntropyLoss()
callbacks=[
    RecallCallback(),
    EarlyStoppingCallback(patience = 7)
    ]

runner.train(
    model=model,
    main_metric='loss',
    minimize_metric=True,
    criterion=criterion,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders=loaders,
    logdir=os.path.join(DATA_FOLDER, './baseline_effnet_one_head/vowel_diacritic'),
    scheduler=scheduler,
    fp16=True,
    num_epochs=50,
    verbose=True
    )

del model

print("_____________Training consonant diacritic____________")

train_dataset = ImageDataset(
    df = image_data.loc[train_image_ids,:],
    labels = all_data.loc[train_mask, :],
    label = 'consonant_diacritic',
    transforms = transforms_train
    )
val_dataset = ImageDataset(
    df = image_data.loc[valid_image_ids,:],
    labels = all_data.loc[val_mask, :],
    label = 'consonant_diacritic',
    transforms = transforms_val
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=True
    )
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=False
    )

model = EfficientNet.from_pretrained("efficientnet-b3", num_classes=7, in_channels=1)
model.cuda()

loaders = collections.OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = val_loader

runner = SupervisedRunner()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
    )

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.1,
    patience=5
    )

criterion = torch.nn.CrossEntropyLoss()
callbacks=[
    RecallCallback(),
    EarlyStoppingCallback(patience = 7)
    ]

runner.train(
    model=model,
    main_metric='loss',
    minimize_metric=True,
    criterion=criterion,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders=loaders,
    logdir=os.path.join(DATA_FOLDER, './baseline_effnet_one_head/consonant_diacritic'),
    scheduler=scheduler,
    fp16=True,
    num_epochs=50,
    verbose=True
    )

del model

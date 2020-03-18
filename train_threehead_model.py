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
from catalyst.dl import Callback, MetricCallback, CallbackOrder, CriterionCallback
import random
import collections
from catalyst.utils import set_global_seed
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, CriterionAggregatorCallback
from catalyst.contrib.nn.optimizers import RAdam, Lookahead, Lamb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet

#from efficientnet.model import EfficientNet
#from efficientnet.utils import get_same_padding_conv2d, round_filters
from resnet.model import resnet50, resnext101_32x8d
import pretrainedmodels
from utils import *
import torch.nn as nn

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

set_global_seed(42)

DATA_FOLDER = '../../data/kaggle'

grapheme_map = pd.read_csv(os.path.join(DATA_FOLDER, 'grapheme_map.csv'))
grapheme_map = dict(zip(grapheme_map["grapheme"], grapheme_map["idx"]))

all_data = pd.read_csv(os.path.join(DATA_FOLDER, './gc_224_oof.csv'))
all_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = all_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')

train_mask = np.bitwise_and(all_data['fold'] != 0, all_data['intopk'])
val_mask = all_data['fold'] == 0

train_image_ids = all_data['image_id'][train_mask].values
valid_image_ids = all_data['image_id'][val_mask].values

ny1 = np.array(all_data.groupby("vowel_diacritic").count()["image_id"].tolist())
ny2 = np.array(all_data.groupby("grapheme_root").count()["image_id"].tolist())
ny3 = np.array(all_data.groupby("consonant_diacritic").count()["image_id"].tolist())

def get_w(ny, beta=0.999):
    return torch.Tensor((1 - beta) / (1 - beta **ny)).cuda()

class ClassificationModel(nn.Module): 
    def __init__(self, 
                 backbone : str, 
                 n_output : int, 
                 input_channels : int = 3, 
                 pretrained : bool =True, 
                 activation=None):
        super(ClassificationModel, self).__init__()
        """
        The aggregation model of different predefined archtecture

        Args:
            backbone : model architecture to use, one of (resnet18 | resnet34 | densenet121 | se_resnext50_32x4d | se_resnext101_32x4d | efficientnet-b0 - efficientnet-b6)
            n_output : number of classes to predict
            input_channels : number of channels for the input image
            pretrained : bool value either to use weights pretrained on imagenet or to random initialization
            activation : a callable will be applied at the very end
        """
        self.backbone = backbone

        if backbone == "resnet18": 
            self.encoder = models.resnet18(pretrained=pretrained) 
        elif backbone == "resnet34": 
            self.encoder = models.resnet34(pretrained=pretrained) 
        elif backbone == "densenet121": 
            self.encoder = models.densenet121(pretrained=pretrained)
        elif backbone == "se_resnext50_32x4d": 
            if pretrained:
                self.encoder = pretrainedmodels.se_resnext50_32x4d(pretrained = 'imagenet') 
            else:
                self.encoder = pretrainedmodels.se_resnext50_32x4d(pretrained = None) 
        elif backbone == "se_resnext101_32x4d":
            if pretrained:
                self.encoder = pretrainedmodels.se_resnext101_32x4d(pretrained = 'imagenet') 
            else:
                self.encoder = pretrainedmodels.se_resnext101_32x4d(pretrained = None) 

        avgpool = nn.AdaptiveAvgPool2d(1)

        if backbone == "resnet34" or backbone == "resnet18":
            if input_channels != 3:
                conv = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                conv.weight.data = self.encoder.conv1.weight.data.sum(dim=1).unsqueeze(1).repeat_interleave(input_channels, dim=1)
                self.encoder.conv1 = conv

            self.encoder.avgpool = avgpool
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, n_output)
        elif backbone == "densenet121": 
            if input_channels != 3:
                conv = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                conv.weight.data = self.encoder.features.conv0.weight.data.sum(dim=1).unsqueeze(1).repeat_interleave(input_channels, dim=1)
                self.encoder.features.conv0 = conv 

            self.encoder.classifier = nn.Linear(self.encoder.classifier.in_features, n_output)
            self.encoder.avgpool = avgpool
        elif backbone == "se_resnext50_32x4d" or backbone == "se_resnext101_32x4d":
            if input_channels != 3:
                conv = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                conv.weight.data = self.encoder.layer0.conv1.weight.data.sum(dim=1).unsqueeze(1).repeat_interleave(input_channels, dim=1)
                self.encoder.layer0.conv1 = conv 

            self.encoder.avg_pool = avgpool
            in_features = self.encoder.last_linear.in_features
            self.encoder.last_linear = nn.Identity()

        elif backbone.startswith("efficientnet"):
            self.encoder = EfficientNet.from_pretrained(backbone, advprop=True)

            if input_channels != 3:
                self.encoder._conv_stem = nn.Conv2d(input_channels, self.encoder._conv_stem.out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
     
            self.encoder._avg_pooling = avgpool
            in_features = self.encoder._fc.in_features
            self.encoder._fc = nn.Identity()

        self.fc0 = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 1024), nn.LeakyReLU(0.1), nn.BatchNorm1d(num_features=1024), nn.Linear(1024, n_output[0]))
        self.fc1 = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 1024), nn.LeakyReLU(0.1), nn.BatchNorm1d(num_features=1024), nn.Linear(1024, n_output[1]))
        self.fc2 = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 1024), nn.LeakyReLU(0.1), nn.BatchNorm1d(num_features=1024), nn.Linear(1024, n_output[2]))
        self.fc3 = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 1024), nn.LeakyReLU(0.1), nn.BatchNorm1d(num_features=1024), nn.Linear(1024, n_output[3]))
        self.activation = activation 


    def forward(self, x):
        x = self.encoder(x) 

        x0, x1, x2, x3 = self.fc0(x), self.fc1(x), self.fc2(x), self.fc3(x)

        return {'logit_vowel_diacritic': x0,
                'logit_grapheme_root': x1,
                'logit_consonant_diacritic': x2,
                'logit_grapheme' : x3}


TRAIN = [os.path.join(DATA_FOLDER, 'train_image_data_' + str(i) + '.parquet') for i in range(4)]
data_full = pd.concat([pd.read_parquet(path) for path in TRAIN],ignore_index=True)
data_train_df = data_full[train_mask]
data_valid_df = data_full[val_mask]

IMG_SIZE = 128
batch_size = 128
num_workers = 16

transforms_train = A.Compose([
        A.Resize(width=IMG_SIZE,
                height=IMG_SIZE),
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
    A.Resize(width=IMG_SIZE,
             height=IMG_SIZE),
    A.Normalize(mean=0.5,
                std=0.5)
])

train_dataset = ImageDataset(
#    df = image_data.loc[train_image_ids,:],
    df = data_train_df,
    labels = all_data.loc[train_mask, :],
    label = 'all',
    grapheme_map = grapheme_map,
    transforms = transforms_train
    )
val_dataset = ImageDataset(
#    df = image_data.loc[valid_image_ids,:],
    df = data_valid_df,
    labels = all_data.loc[val_mask, :],
    label = 'all',
    grapheme_map = grapheme_map,
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

#model = EfficientNet.from_name("efficientnet-b8")
#Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
#out_channels = round_filters(32, model._global_params)
#model._conv_stem = Conv2d(1, out_channels, kernel_size=3, stride=2, bias=False)
#model.load_state_dict(torch.load('effnetb8_three_heavy_head_merge/checkpoints/best.pth')["model_state_dict"], strict=True)



#state_dict = torch.load('resnext50/checkpoints/best_full.pth')
model = ClassificationModel(backbone = "efficientnet-b8", n_output = [11,168,7,1295], input_channels=1)
#model.load_state_dict(state_dict["model_state_dict"], strict=True)
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

#optimizer.load_state_dict(state_dict["optimizer_state_dict"]) 
#for param_group in optimizer.param_groups:
#    param_group['lr'] = 1e-5
            
scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=3, mode='max')

criterions_dict = {
    'vowel_diacritic_loss':torch.nn.CrossEntropyLoss(weight=get_w(ny1)), 
    'grapheme_root_loss':torch.nn.CrossEntropyLoss(weight=get_w(ny2)),
    'consonant_diacritic_loss':torch.nn.CrossEntropyLoss(weight=get_w(ny3)),
    'grapheme_loss' : torch.nn.CrossEntropyLoss()
    }

callbacks=[
    MixupCutmixCallback(fields=["image"], 
                        output_key=("logit_grapheme_root", "logit_vowel_diacritic", "logit_consonant_diacritic", "logit_grapheme"),
                        input_key=("grapheme_root", "vowel_diacritic", "consonant_diacritic", "grapheme"),
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
    main_metric='taskmetric',
    minimize_metric=False,
    criterion=criterions_dict,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders=loaders,
    logdir=os.path.join('./tmp'),
    scheduler=scheduler,
    fp16=True,
    num_epochs=200,
    verbose=True
    )

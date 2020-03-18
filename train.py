import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.augmentations import functional as F
from albumentations import (
    HorizontalFlip,
    ShiftScaleRotate,
    CoarseDropout,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    IAAPiecewiseAffine,
    Cutout,
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

from utils import *
import torch.nn as nn
from argparse import ArgumentParser

parser = ArgumentParser("Bengali training")

parser.add_argument(
    "--cuda_visible_devices", type=str, help="GPUs for training", default="0"
)
parser.add_argument(
    "--data_folder", type=str, help="Path to folder with data", default="data"
)
parser.add_argument(
    "--image_height", type=int, help="Height of images to train", default=137
)
parser.add_argument(
    "--image_width", type=int, help="Width of images to train", default=236
)
parser.add_argument("--batch_size", type=int, help="Batch size", default=128)
parser.add_argument("--num_workers", type=int, help="Number of workers", default=2)
parser.add_argument("--num_epochs", type=int, help="Number of epochs", default=100)
parser.add_argument(
    "--backbone", type=str, help="Model to train", default="se_resnext101_32x4d"
)
parser.add_argument("--lr", type=int, help="Starting learning rate", default=1e-4)
parser.add_argument("--log_path", type=str, help="Path to logs", default="logs")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

set_global_seed(42)

DATA_FOLDER = args.data_folder

grapheme_map = pd.read_csv(os.path.join(DATA_FOLDER, "grapheme_map.csv"))
grapheme_map = dict(zip(grapheme_map["grapheme"], grapheme_map["idx"]))

all_data = pd.read_csv(os.path.join(DATA_FOLDER, "./gc_224_oof.csv"))
all_data[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]] = all_data[
    ["grapheme_root", "vowel_diacritic", "consonant_diacritic"]
].astype("uint8")

train_mask = np.bitwise_and(all_data["fold"] != 0, all_data["intopk"])
val_mask = all_data["fold"] == 0

train_image_ids = all_data["image_id"][train_mask].values
valid_image_ids = all_data["image_id"][val_mask].values

ny1 = np.array(all_data.groupby("vowel_diacritic").count()["image_id"].tolist())
ny2 = np.array(all_data.groupby("grapheme_root").count()["image_id"].tolist())
ny3 = np.array(all_data.groupby("consonant_diacritic").count()["image_id"].tolist())


TRAIN = [
    os.path.join(DATA_FOLDER, "train_image_data_" + str(i) + ".parquet")
    for i in range(4)
]
data_full = pd.concat([pd.read_parquet(path) for path in TRAIN], ignore_index=True)
data_train_df = data_full[train_mask]
data_valid_df = data_full[val_mask]
del data_full

IMG_SIZE = (args.image_height, args.image_width)
batch_size = args.batch_size
num_workers = args.num_workers

transforms_train = A.Compose(
    [
        A.Resize(width=IMG_SIZE[1], height=IMG_SIZE[0]),
        A.OneOf(
            [
                A.RandomContrast(),
                A.RandomBrightness(),
                A.RandomGamma(),
                A.RandomBrightnessContrast(),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.GridDistortion(),
                A.ElasticTransform(),
                A.OpticalDistortion(),
                A.ShiftScaleRotate(),
            ],
            p=0.25,
        ),
        A.CoarseDropout(),
        A.Normalize(mean=0.5, std=0.5),
    ],
    p=1.0,
)

transforms_val = A.Compose(
    [A.Resize(width=IMG_SIZE[1], height=IMG_SIZE[0]), A.Normalize(mean=0.5, std=0.5)]
)

train_dataset = ImageDataset(
    df=data_train_df,
    labels=all_data.loc[train_mask, :],
    label="all",
    grapheme_map=grapheme_map,
    transforms=transforms_train,
)
val_dataset = ImageDataset(
    df=data_valid_df,
    labels=all_data.loc[val_mask, :],
    label="all",
    grapheme_map=grapheme_map,
    transforms=transforms_val,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=False,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=False,
    shuffle=False,
)

model = ClassificationModel(
    backbone=args.backbone, n_output=[11, 168, 7, 1295], input_channels=1
)
model.cuda()

loaders = collections.OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = val_loader

runner = SupervisedRunner(input_key="image", output_key=None, input_target_key=None)

optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=0.001)

scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=3, mode="max")

criterions_dict = {
    "vowel_diacritic_loss": torch.nn.CrossEntropyLoss(weight=get_w(ny1)),
    "grapheme_root_loss": torch.nn.CrossEntropyLoss(weight=get_w(ny2)),
    "consonant_diacritic_loss": torch.nn.CrossEntropyLoss(weight=get_w(ny3)),
    "grapheme_loss": torch.nn.CrossEntropyLoss(),
}

callbacks = [
    MixupCutmixCallback(
        fields=["image"],
        output_key=(
            "logit_grapheme_root",
            "logit_vowel_diacritic",
            "logit_consonant_diacritic",
            "logit_grapheme",
        ),
        input_key=(
            "grapheme_root",
            "vowel_diacritic",
            "consonant_diacritic",
            "grapheme",
        ),
        mixuponly=False,
        alpha=0.5,
        resolution=IMG_SIZE,
    ),
    CriterionCallback(
        input_key="grapheme_root",
        output_key="logit_grapheme_root",
        prefix="grapheme_root_loss",
        criterion_key="grapheme_root_loss",
        multiplier=2.0,
    ),
    CriterionCallback(
        input_key="vowel_diacritic",
        output_key="logit_vowel_diacritic",
        prefix="vowel_diacritic_loss",
        criterion_key="vowel_diacritic_loss",
        multiplier=1.0,
    ),
    CriterionCallback(
        input_key="consonant_diacritic",
        output_key="logit_consonant_diacritic",
        prefix="consonant_diacritic_loss",
        criterion_key="consonant_diacritic_loss",
        multiplier=1.0,
    ),
    CriterionAggregatorCallback(
        prefix="loss",
        loss_keys=[
            "grapheme_root_loss",
            "vowel_diacritic_loss",
            "consonant_diacritic_loss",
        ],
    ),
    TaskMetricCallback(
        output_key=(
            "logit_grapheme_root",
            "logit_vowel_diacritic",
            "logit_consonant_diacritic",
        ),
        input_key=("grapheme_root", "vowel_diacritic", "consonant_diacritic"),
    ),
]

runner.train(
    model=model,
    main_metric="taskmetric",
    minimize_metric=False,
    criterion=criterions_dict,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders=loaders,
    logdir=args.log_path,
    scheduler=scheduler,
    fp16=True,
    num_epochs=args.num_epochs,
    verbose=True,
)

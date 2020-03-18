from tqdm import tqdm
from math import ceil
import pandas as pd
import numpy as np
import os

os.system(
    "pip install ../input/bengalimodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/"
)

from torch import nn, from_numpy, no_grad, load
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import pretrainedmodels

BATCH_SIZE = 256
IMG_SIZE = 224

models = {
    "../input/bengalimodels/224_0_se_resnext101_32x4d.pth": {
        "backbone": "se_resnext101_32x4d",
        "model": None,
    },
    "../input/bengalimodels/224_1_se_resnext101_32x4d.pth": {
        "backbone": "se_resnext101_32x4d",
        "model": None,
    },
    "../input/bengalimodels/224_2_se_resnext101_32x4d.pth": {
        "backbone": "se_resnext101_32x4d",
        "model": None,
    },
    "../input/bengalimodels/224_3_se_resnext101_32x4d.pth": {
        "backbone": "se_resnext101_32x4d",
        "model": None,
    },
    "../input/bengalimodels/224_4_se_resnext101_32x4d.pth": {
        "backbone": "se_resnext101_32x4d",
        "model": None,
    },
}


class ClassificationModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        n_output: int,
        input_channels: int = 3,
        pretrained: bool = True,
        activation=None,
    ):
        super(ClassificationModel, self).__init__()
        """
        The aggregation model of different predefined archtecture

        Args:
            backbone : model architecture to use, one of (resnet18 | resnet34 | densenet121 | se_resnext50_32x4d | se_resnext101_32x4d)
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
                self.encoder = pretrainedmodels.se_resnext50_32x4d(
                    pretrained="imagenet"
                )
            else:
                self.encoder = pretrainedmodels.se_resnext50_32x4d(pretrained=None)
        elif backbone == "se_resnext101_32x4d":
            if pretrained:
                self.encoder = pretrainedmodels.se_resnext101_32x4d(
                    pretrained="imagenet"
                )
            else:
                self.encoder = pretrainedmodels.se_resnext101_32x4d(pretrained=None)

        avgpool = nn.AdaptiveAvgPool2d(1)

        if backbone == "resnet34" or backbone == "resnet18":
            if input_channels != 3:
                conv = nn.Conv2d(
                    input_channels,
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
                conv.weight.data = (
                    self.encoder.conv1.weight.data.sum(dim=1)
                    .unsqueeze(1)
                    .repeat_interleave(input_channels, dim=1)
                )
                self.encoder.conv1 = conv
            self.encoder.avgpool = avgpool
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, n_output)
        elif backbone == "densenet121":
            if input_channels != 3:
                conv = nn.Conv2d(
                    input_channels,
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
                conv.weight.data = (
                    self.encoder.features.conv0.weight.data.sum(dim=1)
                    .unsqueeze(1)
                    .repeat_interleave(input_channels, dim=1)
                )
                self.encoder.features.conv0 = conv
            self.encoder.classifier = nn.Linear(
                self.encoder.classifier.in_features, n_output
            )
            self.encoder.avgpool = avgpool
        elif backbone == "se_resnext50_32x4d" or backbone == "se_resnext101_32x4d":
            if input_channels != 3:
                conv = nn.Conv2d(
                    input_channels,
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
                conv.weight.data = (
                    self.encoder.layer0.conv1.weight.data.sum(dim=1)
                    .unsqueeze(1)
                    .repeat_interleave(input_channels, dim=1)
                )
                self.encoder.layer0.conv1 = conv
            self.encoder.avg_pool = avgpool

        in_features = self.encoder.last_linear.in_features

        self.fc0 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(1024, n_output[0]),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(1024, n_output[1]),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(1024, n_output[2]),
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(1024, n_output[3]),
        )
        self.encoder.last_linear = nn.Identity()
        self.activation = activation

    def forward(self, x):
        x = self.encoder(x)

        x0, x1, x2, x3 = self.fc0(x), self.fc1(x), self.fc2(x), self.fc3(x)

        return {
            "logit_vowel_diacritic": x0,
            "logit_grapheme_root": x1,
            "logit_consonant_diacritic": x2,
            "logit_grapheme": x3,
        }


class AlphabetDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.data = df.iloc[:, 1:].values
        self.transforms = transforms

    def __getitem__(self, idx):
        flattened_image = self.data[idx, :].astype(np.uint8)
        image = np.expand_dims(flattened_image.reshape(137, 236), 2)
        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        image = from_numpy(image.transpose((2, 0, 1)))
        return image

    def __len__(self):
        return len(self.data)


transforms_val = A.Compose(
    [A.Resize(width=IMG_SIZE, height=IMG_SIZE), A.Normalize(mean=0.5, std=0.5)]
)


for name in models.keys():
    models[name]["model"] = ClassificationModel(
        backbone=models[name]["backbone"],
        n_output=[11, 168, 7, 1295],
        input_channels=1,
        pretrained=False,
    )
    models[name]["model"].load_state_dict(load(name)["model_state_dict"])
    models[name]["model"].cuda()
    models[name]["model"].eval()


test_data = [
    "test_image_data_0.parquet",
    "test_image_data_1.parquet",
    "test_image_data_2.parquet",
    "test_image_data_3.parquet",
]
predictions = {"grapheme_root": [], "vowel_diacritic": [], "consonant_diacritic": []}

for fname in test_data:
    data = pd.read_parquet(f"../input/bengaliai-cv19/{fname}")
    infere_dataset = AlphabetDataset(df=data, transforms=transforms_val)
    infere_loader = DataLoader(
        infere_dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
    )
    iterator = iter(infere_loader)
    with no_grad():
        for step in tqdm(range(ceil(len(infere_dataset) / BATCH_SIZE))):
            image_batch = next(iterator)
            image_batch = image_batch.float().cuda()

            preds = [models[fold]["model"](image_batch) for fold in models.keys()]

            for i in range(1, len(preds)):
                preds[0]["logit_consonant_diacritic"] += preds[i][
                    "logit_consonant_diacritic"
                ]
                preds[0]["logit_grapheme_root"] += preds[i]["logit_grapheme_root"]
                preds[0]["logit_vowel_diacritic"] += preds[i]["logit_vowel_diacritic"]

            predictions["consonant_diacritic"].extend(
                np.argmax(preds[0]["logit_consonant_diacritic"].cpu().numpy(), axis=1)
            )
            predictions["grapheme_root"].extend(
                np.argmax(preds[0]["logit_grapheme_root"].cpu().numpy(), axis=1)
            )
            predictions["vowel_diacritic"].extend(
                np.argmax(preds[0]["logit_vowel_diacritic"].cpu().numpy(), axis=1)
            )

answer = np.array(
    [
        predictions["consonant_diacritic"],
        predictions["grapheme_root"],
        predictions["vowel_diacritic"],
    ]
)

submission = pd.read_csv("../input/bengaliai-cv19/sample_submission.csv")
submission.target = np.hstack(answer.T)
submission.to_csv("submission.csv", index=False)

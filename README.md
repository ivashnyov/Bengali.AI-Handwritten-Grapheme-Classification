# Bengali.AI-Handwritten-Grapheme-Classification
Our team **[ods.ai] Here we go again** took 46th place in this competition. This is our training pipeline for this task.
# Approach
We used the following tricks:

- Four-headed models
- Augmentations
- Cutmix
- Mixup

There is nothing innovative here, we just did it carefully :wink:
# Final submission
Our final submission consists of four **se_resnext101_32x4d** models, trained on four different folds for 200 epochs on 128x128 image size and then trained for 200 epochs on 224x224 image size with a smaller learning rate.
# Appreciation
- [Catalyst](https://github.com/catalyst-team/catalyst)
- [Albumentations](https://github.com/albumentations-team/albumentations)


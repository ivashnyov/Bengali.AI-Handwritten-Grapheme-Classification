import random
import numpy as np
import torch
from torch.nn.modules import Module
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
from torch.utils.data import Dataset
from catalyst.dl import Callback, MetricCallback, CallbackOrder, CriterionCallback, State
from sklearn.metrics import recall_score
from typing import List

class GridDropout(DualTransform):
    """
    GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.
        Args:
            ratio (float): the ratio of the mask holes to the unit_size (same for horizontal and vertical directions).
                Must be between 0 and 1. Default: 0.5.
            unit_size_min (int): minimum size of the grid unit. Must be between 2 and the image shorter edge.
                If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
            unit_size_max (int): maximum size of the grid unit. Must be between 2 and the image shorter edge.
                If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
            holes_number_x (int): the number of grid units in x direction. Must be between 1 and image width//2.
                If 'None', grid unit width is set as image_width//10. Default: `None`.
            holes_number_y (int): the number of grid units in y direction. Must be between 1 and image height//2.
                If `None`, grid unit height is set equal to the grid unit width or image height, whatever is smaller.
            shift_x (int): offsets of the grid start in x direction from (0,0) coordinate.
                Clipped between 0 and grid unit_width - hole_width. Default: 0.
            shift_y (int): offsets of the grid start in y direction from (0,0) coordinate.
                Clipped between 0 and grid unit_width - hole_width. Default: 0.
            shift_y (int): offsets of the grid start in y direction from (0,0) coordinate.
                Clipped between 0 and grid unit height - hole_height. Default: 0.
            random_offset (boolean): weather to offset the grid randomly between 0 and grid unit size - hole size
                If 'True', entered shift_x, shift_y are ignored and set randomly. Default: `False`.
            fill_value (int): value for the dropped pixels. Default = 0
            mask_fill_value (int): value for the dropped pixels in mask.
                If `None`, tranformation is not applied to the mask. Default: `None`.
        Targets:
            image, mask
        Image types:
            uint8, float32
        References:
            https://arxiv.org/abs/2001.04086
    """

    def __init__(
            self,
            ratio: float = 0.5,
            unit_size_min: int = None,
            unit_size_max: int = None,
            holes_number_x: int = None,
            holes_number_y: int = None,
            shift_x: int = 0,
            shift_y: int = 0,
            random_offset: bool = False,
            fill_value: int = 0,
            mask_fill_value: int = None,
            always_apply: bool = False,
            p: float = 0.5,
    ):
        super(GridDropout, self).__init__(always_apply, p)
        self.ratio = ratio
        self.unit_size_min = unit_size_min
        self.unit_size_max = unit_size_max
        self.holes_number_x = holes_number_x
        self.holes_number_y = holes_number_y
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.random_offset = random_offset
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        if not 0 < self.ratio <= 1:
            raise ValueError("ratio must be between 0 and 1.")

    def apply(self, image, holes=[], **params):
        return F.cutout(image, holes, self.fill_value)

    def apply_to_mask(self, image, holes=[], **params):
        if self.mask_fill_value is None:
            return image
        else:
            return F.cutout(image, holes, self.mask_fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]
        # set grid using unit size limits
        if self.unit_size_min and self.unit_size_max:
            if not 2 <= self.unit_size_min <= self.unit_size_max:
                raise ValueError("Max unit size should be >= min size, both at least 2 pixels.")
            if self.unit_size_max > min(height, width):
                raise ValueError("Grid size limits must be within the shortest image edge.")
            unit_width = random.randint(self.unit_size_min, self.unit_size_max + 1)
            unit_height = unit_width
        else:
            # set grid using holes numbers
            if self.holes_number_x is None:
                unit_width = max(2, width // 10)
            else:
                if not 1 <= self.holes_number_x <= width // 2:
                    raise ValueError("The hole_number_x must be between 1 and image width//2.")
                unit_width = width // self.holes_number_x
            if self.holes_number_y is None:
                unit_height = max(min(unit_width, height), 2)
            else:
                if not 1 <= self.holes_number_y <= height // 2:
                    raise ValueError("The hole_number_y must be between 1 and image height//2.")
                unit_height = height // self.holes_number_y

        hole_width = int(unit_width * self.ratio)
        hole_height = int(unit_height * self.ratio)
        # min 1 pixel and max unit length - 1
        hole_width = min(max(hole_width, 1), unit_width - 1)
        hole_height = min(max(hole_height, 1), unit_height - 1)
        # set offset of the grid
        if self.shift_x is None:
            shift_x = 0
        else:
            shift_x = min(max(0, self.shift_x), unit_width - hole_width)
        if self.shift_y is None:
            shift_y = 0
        else:
            shift_y = min(max(0, self.shift_y), unit_height - hole_height)
        if self.random_offset:
            shift_x = random.randint(0, unit_width - hole_width)
            shift_y = random.randint(0, unit_height - hole_height)
        holes = []
        for i in range(width // unit_width + 1):
            for j in range(height // unit_height + 1):
                x1 = min(shift_x + unit_width * i, width)
                y1 = min(shift_y + unit_height * j, height)
                x2 = min(x1 + hole_width, width)
                y2 = min(y1 + hole_height, height)
                holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "ratio",
            "unit_size_min",
            "unit_size_max",
            "holes_number_x",
            "holes_number_y",
            "shift_x",
            "shift_y",
            "mask_fill_value",
            "random_offset",
        )


class AugMix(DualTransform):
    """Augmentations mix to Improve Robustness and Uncertainty.

    Args:
        image (np.ndarray): Raw input image of shape (h, w, c)
        severity (int): Severity of underlying augmentation operators.
        width (int): Width of augmentation chain
        depth (int): Depth of augmentation chain. -1 enables stochastic depth uniformly
          from [1, 3]
        alpha (float): Probability coefficient for Beta and Dirichlet distributions.
        augmentations (list of augmentations): Augmentations that need to mix and perform.

    Target:
        image

    Image types:
        uint8, float32

    Returns:
        mixed: Augmented and mixed image.

    Reference:
    |   https://arxiv.org/abs/1912.02781
    |   https://github.com/google-research/augmix
    """

    def __init__(self, width=4,
                 depth=3,
                 alpha=0.5,
                 augmentations=None,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 always_apply=False,
                 resize_width=None,
                 resize_height=None,
                 p=0.5):
        super(AugMix, self).__init__(always_apply, p)
        if isinstance(augmentations, (list, tuple)):
            self.augmentations = augmentations
        else:
            raise ValueError("Augmentations list should be passed to 'augmentations' argument.")
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.mean = mean
        self.std = std
        self.resize_width = resize_width
        self.resize_height = resize_height

    def apply_op(self, image, op):
        image = np.clip(image * 255., 0, 255).astype(np.uint8) \
            if 'float32' not in op.__doc__ \
            else image
        image = op(image=image)['image']
        return image

    def apply(self, img, **params):
        ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))

        mix = np.float32(np.zeros_like(img))
        for i in range(self.width):
            image_aug = img.copy()

            for _ in range(self.depth):
                op = np.random.choice(self.augmentations)
                image_aug = self.apply_op(img, op)

        # Preprocessing commutes since all coefficients are convex
        mix = np.add(mix, ws[i] * F.normalize(image_aug, mean=self.mean, std=self.std), out=mix, casting="unsafe")
        mixed = (1 - m) * F.normalize(img, mean=self.mean, std=self.std) + m * mix
        if self.resize_height is not None and self.resize_width is not None:
            mixed = F.resize(mixed, height=self.resize_height, width=self.resize_width)
        return mixed

    def get_transform_init_args_names(self):
        return ('width', 'depth', 'alpha', 'mean', 'std', 'height', 'width')


class ImageDataset(Dataset):
    def __init__(self,
                 df,
                 labels,
                 label,
                 grapheme_map,
                 transforms=None):
        self.df = df.copy()
        self.labels = labels.copy()
        self.label = label
        self.transforms = transforms
        self.grapheme_map = grapheme_map

    def __getitem__(self, idx):
        flattened_image = self.df.iloc[idx, 1:].values.astype(np.uint8)
        image = np.expand_dims(flattened_image.reshape(137, 236), 2)

        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented['image']

        if self.label == 'grapheme_root':
            label = self.labels['grapheme_root'].values[idx]
        elif self.label == 'vowel_diacritic':
            label = self.labels['vowel_diacritic'].values[idx]
        elif self.label == 'consonant_diacritic':
            label = self.labels['consonant_diacritic'].values[idx]
        elif self.label == 'all':
            grapheme_root =  self.labels['grapheme_root'].values[idx]
            vowel_diacritic = self.labels['vowel_diacritic'].values[idx]
            consonant_diacritic = self.labels['consonant_diacritic'].values[idx]
            grapheme = self.grapheme_map[self.labels['grapheme'].values[idx]]

            image = torch.from_numpy(image.transpose((2, 0, 1)))
            grapheme_root = torch.tensor(grapheme_root).long()
            vowel_diacritic = torch.tensor(vowel_diacritic).long()
            consonant_diacritic = torch.tensor(consonant_diacritic).long() 
            
            output_dict  = {
                'grapheme_root' : grapheme_root, 
                'vowel_diacritic' : vowel_diacritic, 
                'consonant_diacritic' : consonant_diacritic,
                'grapheme' : grapheme, 
                'image' : image
                        }

            return output_dict

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        label = torch.tensor(label).long()

        return image, label

    def __len__(self):
        return len(self.df)


def recall(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = None,
        activation: str = None
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: recall
    """
    outputs = torch.argmax(outputs, dim=1)
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    score = recall_score(targets, outputs, average='macro')
    return score


class RecallCallback(MetricCallback):
    """
    Recall score metric callback.
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "recall",
            activation: str = None
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """

        super().__init__(
            prefix=prefix,
            metric_fn=recall,
            input_key=input_key,
            output_key=output_key,
            activation=activation
        )


class TaskMetricCallback(Callback):
    '''
    Proposed metrics:
    import numpy as np
    import sklearn.metrics

    scores = []
    for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
        y_true_subset = solution[solution[component] == component]['target'].values
        y_pred_subset = submission[submission[component] == component]['target'].values
        scores.append(sklearn.metrics.recall_score(
            y_true_subset, y_pred_subset, average='macro'))
    final_score = np.average(scores, weights=[2,1,1])
    '''

    def __init__(
        self, 
        input_key: str = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic'], 
        output_key: str = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic'],
        class_names: str = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic'],
        prefix: str = "taskmetric", 
        ignore_index=None
    ):
        super().__init__(CallbackOrder.Metric)
        self.metric_fn = lambda outputs, targets: recall_score(targets, outputs, average="macro")
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.class_names = class_names
        self.outputs = [[] for i in range(3)]
        self.targets = [[] for i in range(3)]

    def on_batch_end(self, state: State):
        
        for i in range(3):
            outputs = state.output[self.output_key[i]].detach().cpu().numpy()
            targets = state.input[self.input_key[i]].detach().cpu().numpy()
            #num_classes = outputs.shape[1]
            outputs = np.argmax(outputs, axis=1)
            #outputs = [np.eye(num_classes)[y] for y in outputs]
            #targets = [np.eye(num_classes)[y] for y in targets]
            self.outputs[i].extend(outputs)
            self.targets[i].extend(targets)

    def on_loader_start(self, state):
        self.outputs = [[] for i in range(3)]
        self.targets = [[] for i in range(3)]

    def on_loader_end(self, state):
        metric_name = self.prefix
        score_vec = []
        for i in range(3):
            targets = np.array(self.targets[i])
            outputs = np.array(self.outputs[i])
            metric = self.metric_fn(outputs, targets)
            score_vec.append(metric)
            state.metric_manager.epoch_values[state.loader_name][self.class_names[i]] = float(metric)
            
            
        state.metric_manager.epoch_values[state.loader_name][metric_name] = np.average(score_vec, weights=[2,1,1])

def get_rand_mask(img_size):
    sigma = np.random.randint(25, 30)
    r = sigma + np.random.randint(-20, -10)
    shift = np.random.randint(0, 20)

    total = sigma + r

    tile_number = 1 + img_size // total

    mask = np.zeros((sigma + r, sigma + r))
    mask[:sigma, :sigma] = 1

    final_mask = np.zeros((img_size, img_size))


    final_mask[shift:, shift:] = np.tile(mask, (tile_number, tile_number))[:img_size - shift, :img_size - shift]
    return final_mask

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class MixupCutmixCallback(CriterionCallback):
    def __init__(
            self,
            fields: List[str] = ("image",),
            alpha=1.0,
            on_train_only=True,
            weight_grapheme_root=2.0,
            weight_vowel_diacritic=1.0,
            weight_consonant_diacritic=1.0,
            mixuponly=True,
            **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, \
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"
        super().__init__(**kwargs)
        print("Custom MixupCutmixCallback is being initialized!")
        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True
        self.weight_grapheme_root = weight_grapheme_root
        self.weight_vowel_diacritic = weight_vowel_diacritic
        self.weight_consonant_diacritic = weight_consonant_diacritic
        self.apply_mixup = True
        self.mixuponly = mixuponly
        self.mixup_prob = 0.7
        self.cutmix_prob = 0.1

    def on_loader_start(self, state: State):
        self.is_needed = not self.on_train_only or \
                         state.loader_name.startswith("train")

    def do_mixup(self, state: State):
        for f in self.fields:
            state.input[f] = self.lam * state.input[f] + \
                             (1 - self.lam) * state.input[f][self.index]

    def do_cutmix(self, state: State):
        bbx1, bby1, bbx2, bby2 =\
            rand_bbox(state.input[self.fields[0]].shape, self.lam)
        for f in self.fields:
            state.input[f][:, :, bbx1:bbx2, bby1:bby2] =\
                state.input[f][self.index, :, bbx1:bbx2, bby1:bby2]
        self.lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)
                        / (state.input[self.fields[0]].shape[-1]
                           * state.input[self.fields[0]].shape[-2]))

    def on_batch_start(self, state: State):
        if not self.is_needed:
            return
        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1
        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)
        if self.mixuponly:
            self.do_mixup(state)
        else:
            self.r = np.random.rand()
            if self.r < self.mixup_prob:
                self.do_cutmix(state)
            elif self.r < self.mixup_prob + self.cutmix_prob:
                self.do_mixup(state)
            else:
                for i in range(len(state.input)):
                    state.input[self.fields[0]][i] = state.input[self.fields[0]][i] * torch.Tensor(get_rand_mask(128)).cuda()
                

    def _compute_loss(self, state: State, criterion):
        loss_arr = [0, 0, 0, 0]
        if not self.is_needed or self.r >= self.mixup_prob + self.cutmix_prob:
            for i, (input_key, output_key) in enumerate(list(zip(self.input_key, self.output_key))):
                pred = state.output[output_key]
                y = state.input[input_key]
                _criterion = criterion[input_key + '_loss']
                loss_arr[i] = _criterion(pred, y)
        else:
            for i, (input_key, output_key) in enumerate(list(zip(self.input_key, self.output_key))):
                pred = state.output[output_key]
                y_a = state.input[input_key]
                y_b = state.input[input_key][self.index]
                _criterion = criterion[input_key + '_loss']
                loss_arr[i] = self.lam * _criterion(pred, y_a) + \
                              (1 - self.lam) * _criterion(pred, y_b)
        loss = loss_arr[0] * self.weight_grapheme_root + \
               loss_arr[1] * self.weight_vowel_diacritic + \
               loss_arr[2] * self.weight_consonant_diacritic + \
               loss_arr[3] * 1
        return loss


class FocalLoss(Module):
    def __init__(self, gamma=0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = torch.nn.functional.cross_entropy(input, target, reduction=self.reduction)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

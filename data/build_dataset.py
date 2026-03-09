import os
import sys
import torch
import torch.nn as nn
import random
import torchvision
import numpy as np
from PIL import Image
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import ToPILImage
import pickle
import math
from clip import load, tokenize
from clip.custom_clip import DOWNLOAD_ROOT, TextEncoder

from .bird200 import Cub2011
from .food101 import Food101
from .car196 import StanfordCars
from .pet37 import OxfordIIITPet

from .imagenet_prompts import imagenet_templates
from .dataset_class_names import get_classnames


ID_to_DIRNAME={
    'I': 'imagenet_1k',
    'A': 'imagenet-adversarial/imagenet-a',
    'K': 'ImageNet-Sketch/sketch',
    'R': 'imagenet-rendition/imagenet-r',
    'V': 'imagenetv2/imagenetv2-matched-frequency-format-val',
    
    # noisy datasets below
    'LSUN': 'LSUN',
    'Texture': 'dtd/images_file',
    'Places': 'Places',
    'iNaturalist': 'iNaturalist',
    'SUN': 'SUN',
    'ninco': 'ninco',
    'ssb_hard': 'ssb_hard',
}


class CIFAR10(datasets.CIFAR10):
    def __init__(self, *args, is_carry_index=False, **kwargs):
        self.is_carry_index = is_carry_index
        super().__init__(*args, **kwargs)
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        if self.is_carry_index:
            if type(image) == list:
                image.append(index)
            else:
                image = [image, index]
        return image, target


class CIFAR100(datasets.CIFAR100):
    def __init__(self, *args, is_carry_index=False, **kwargs):
        self.is_carry_index = is_carry_index
        super().__init__(*args, **kwargs)
        return
    
    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        if self.is_carry_index:
            if type(image) == list:
                image.append(index)
            else:
                image = [image, index]
        return image, target
import math
import torch
from torchvision.transforms import ToPILImage

class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, transform, data_size, fixed_target=-1000, ratio=1,
                 is_carry_index=False, noise_type='gaussian',
                 num_gaussian=100,
                 select_strategy='by_index',   # 'by_index' 或 'random'
                 use_family_id_as_target=False,
                 normalize_mode='global_uint8' # 'global_uint8' | 'none' | 'per_image_minmax(不建议)'
                 ):
        self.noise_type = noise_type
        self.number = int(data_size * ratio)
        self.fixed_target = fixed_target
        self.transform = transform
        self.is_carry_index = is_carry_index
        self.to_pil = ToPILImage()

        self.num_gaussian = num_gaussian
        self.select_strategy = select_strategy
        self.use_family_id_as_target = use_family_id_as_target
        self.normalize_mode = normalize_mode

        if self.noise_type == 'gaussian':
            self._build_gaussian_families()

    def _build_gaussian_families(self):
        K, C = self.num_gaussian, 3
        means = torch.empty(K, C, 1, 1)
        stds  = torch.empty(K, C, 1, 1)
        for k in range(K):
            means[k].uniform_(-1.0, 1.0)                          # 每组均值 ~ U(-1,1)
            stds[k].log_normal_(mean=math.log(0.5), std=0.4)      # 每组σ ~ LogNormal
        self.gauss_means = means
        self.gauss_stds = stds

    def _choose_family(self, index: int) -> int:
        if self.select_strategy == 'by_index':
            return index % self.num_gaussian
        elif self.select_strategy == 'random':
            return torch.randint(self.num_gaussian, (1,)).item()
        else:
            raise ValueError("select_strategy must be 'by_index' or 'random'")

    def __getitem__(self, index: int):
        chosen_k = None
        chosen_mean = None
        chosen_std = None

        # -------- 生成噪声 --------
        if self.noise_type == 'gaussian':
            chosen_k = self._choose_family(index)
            chosen_mean = self.gauss_means[chosen_k]  # (C,1,1)
            chosen_std  = self.gauss_stds[chosen_k]   # (C,1,1)
            image = torch.randn(3, 224, 224) * chosen_std + chosen_mean

        elif self.noise_type == 'uniform':
            image = torch.rand(3, 224, 224)

        elif self.noise_type == 'salt_and_pepper':
            image = self._salt_and_pepper_noise(torch.zeros(3, 224, 224))

        elif self.noise_type == 'poisson':
            image = self._poisson_noise(torch.ones(3, 224, 224))

        else:
            raise NotImplementedError

        # -------- 目标标签 --------
        if self.use_family_id_as_target and self.noise_type == 'gaussian':
            target = chosen_k
        else:
            target = self.fixed_target

        # -------- 可视化/transform 前的数值处理 --------
        if self.transform is not None:
            if self.normalize_mode == 'global_uint8':
                # 全局固定：clip 到 [-3,3]，统一缩放到 [0,255]（不会按组居中到 127.5）
                img_vis = image.clamp(-3, 3)
                img_vis = ((img_vis + 3.0) / 6.0) * 255.0
                img_vis = img_vis.round().clamp(0, 255).to(torch.uint8)
                img_pil = self.to_pil(img_vis)
                image = self.transform(img_pil)

            elif self.normalize_mode == 'none':
                # 返回原始浮点，不做 0-255 映射；要求 transform 支持 Tensor
                #（如果 transform 需要 PIL，请改用 'global_uint8' 或在外面自己处理）
                image = self.transform(image)

            elif self.normalize_mode == 'per_image_minmax':
                # 不建议：会抹平分布差异（导致余弦很大）
                img_vis = (image - image.min()) / (image.max() - image.min() + 1e-12) * 255.0
                img_vis = img_vis.round().clamp(0, 255).to(torch.uint8)
                img_pil = self.to_pil(img_vis)
                image = self.transform(img_pil)
            else:
                raise ValueError("normalize_mode must be 'global_uint8', 'none', or 'per_image_minmax'")

        if self.is_carry_index:
            if isinstance(image, list):
                image.append(index)
            else:
                image = [image, index]

        return image, target

    def _salt_and_pepper_noise(self, image):
        prob = 0.05
        rnd = torch.rand(image.shape)
        salt = (rnd < prob/2).float()
        pepper = ((rnd >= prob/2) & (rnd < prob)).float()
        image = image * ((~salt.bool()) & (~pepper.bool())).float() + salt * 255
        return image

    def _poisson_noise(self, image):
        noise_param = 1.0
        scaled_image = image * noise_param
        noisy_image = torch.poisson(scaled_image)
        noisy_image = noisy_image / noise_param
        return noisy_image

    def __len__(self):
        return self.number



class FixedOODTargetDataset(torch.utils.data.Dataset):
    """A dataset wrapper that sets a fixed target value for all samples."""
    def __init__(self, dataset, fixed_target):
        self.dataset = dataset
        self.fixed_target = fixed_target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, _ = self.dataset[idx]  # Ignore original target
        return data, self.fixed_target  # Return data with fixed target


class ResampleDataset(torch.utils.data.Dataset):
    """Resample dataset at specified indices."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def get_resampled_indices(dataset, size):
    n = len(dataset)
    if size <= n:
        indices = torch.randperm(n)[:size].tolist()
    else:
        repeat_indices = (size // n) * list(range(n))
        remainder_indices = torch.randperm(n)[:size % n].tolist()
        indices = repeat_indices + remainder_indices
    return indices


def load_id_dataset(set_id, args, transform, corrupt_transform):        
    # NOTE: Since the dataset size of bird200, car196 and pet37 are limited, we use both the training and the test dataset.
    dataset_loaders = {
        'CIFAR-10': lambda: datasets.CIFAR10(root=args.data.path, transform=transform, download=True, train=False),
        'CIFAR-100': lambda: datasets.CIFAR100(root=args.data.path, transform=transform, download=True, train=False),
        'bird200': lambda: Cub2011(root=args.data.path, transform=transform),
        'car196': lambda: StanfordCars(root=args.data.path, download=True, transform=transform),
        'food101': lambda: Food101(root=args.data.path, split="test", download=True, transform=transform),
        'pet37': lambda: OxfordIIITPet(root=args.data.path, splits=["trainval", "test"], download=True, transform=transform),
    }

    if set_id in ['I', 'A', 'K', 'R', 'V']:
        id_testdir = os.path.join(args.data.path, ID_to_DIRNAME[set_id])
        if set_id == 'I':
            id_testdir = os.path.join(id_testdir, 'val_file')
        return datasets.ImageFolder(id_testdir, transform=transform)

    if set_id in dataset_loaders:
        return dataset_loaders[set_id]()
    
    raise ValueError(f"No dataset loader defined for set_id: {set_id}")


def build_test_data(args, transform, corrupt_transform, ttda=False):
    # load ID dataset below
    try:
        id_teset = load_id_dataset(args.data.test_set, args, transform, corrupt_transform)
    except ValueError as e:
        print(e)
    
    id_size = len(id_teset)
    # load noisy dataset below
    if args.data.OOD_set == 'None': # clean data stream
        teset = id_teset
    else: # noisy data stream
        if args.data.OOD_set =='SVHN':
            ood_raw_teset = datasets.SVHN(root=os.path.join(args.data.path, "svhn"), split="test", 
                                       download=True, transform=transform)
        elif args.data.OOD_set =='CIFAR-100':
            ood_raw_teset = load_id_dataset('CIFAR-100', args, transform, corrupt_transform)
        elif args.data.OOD_set in ['LSUN', 'Texture', 'Places', 'iNaturalist', 'SUN', 'ssb_hard', 'ninco']: 
            ood_testdir = os.path.join(args.data.path, ID_to_DIRNAME[args.data.OOD_set])
            ood_raw_teset = datasets.ImageFolder(ood_testdir, transform=transform)
        else:
            raise ValueError(f"Noise dataset {args.data.OOD_set} not recognized.")

        ood_teset = FixedOODTargetDataset(dataset=ood_raw_teset, fixed_target=args.class_num)
        noisy_idx_dir = f'data/noisy_data_idx/{args.data.OOD_ratio}'
        os.makedirs(noisy_idx_dir, exist_ok=True)
        noisy_idx_dir_pkl = f'{noisy_idx_dir}/{args.data.test_set}_{args.data.OOD_set}.pkl'
        if os.path.exists(noisy_idx_dir_pkl):
            with open(noisy_idx_dir_pkl, "rb") as f:
                resampled_indices = pickle.load(f)
        else:
            assert args.method == 'ZS-CLIP' # only support ZS-CLIP to create noisy_idx_dir_pkl to ensure consistency
            resampled_indices = get_resampled_indices(ood_teset, int(id_size * args.data.OOD_ratio))
            with open(noisy_idx_dir_pkl, 'wb') as f:
                pickle.dump(resampled_indices, f)
        ood_teset = ResampleDataset(ood_teset, resampled_indices)

        if args.method == 'ZS-NTTA':
            # 0.125 -> 8, 0.25 -> 4, 0.5 -> 2, 1.0 -> 1
            noise_size = (len(id_teset) + len(ood_teset)) * args.inference.gaussian_rate
            if args.inference.gaussian_rate != 0.125:
                assert args.logs.experiment_group == "ablation_gaussian_rate"
            combined_teset = torch.utils.data.ConcatDataset([id_teset, ood_teset])
            noise_data = NoiseDataset(transform=transform, data_size=noise_size, fixed_target=-1000, is_carry_index=False, noise_type=args.inference.inject_noise_type)
            teset = torch.utils.data.ConcatDataset([combined_teset, noise_data])
        else:
            teset = torch.utils.data.ConcatDataset([id_teset, ood_teset])
    return teset, id_teset


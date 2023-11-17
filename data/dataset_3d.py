import sys
import os
import random
import h5py
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset, Subset, random_split
from torchvision.transforms import Compose, ToPILImage
import torch.fft as fft
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
from copy import deepcopy
from builtins import range


class LADataset(Dataset):
    def __init__(self, image_path='', image_size=[80, 112, 112], stage='train', is_augmentation=False):
        super(LADataset, self).__init__()
        self.image_size = image_size
        self.sep = '\\' if sys.platform[:3] == 'win' else '/'
        self.stage = stage
        self.is_augmentation = is_augmentation
        self.image_path = image_path
        if self.stage == 'train':
            with open(self.image_path + "/train.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        else:
            with open(self.image_path + "/test.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        self.pre_transform = self.pre_transform()
        self.augmentation = self.aug_transform()
        self.post_transform = self.post_transform()

    def __getitem__(self, item):
        image_name = self.sample_list[item]
        h5f = h5py.File(self.image_path + "/LA_h5/" + image_name + "/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.stage == 'train':
            sample = self.pre_transform(sample)
            sampleA1, sampleA2 = deepcopy(sample), deepcopy(sample)
            sampleA1, sampleA2 = self.augmentation(sampleA1), self.augmentation(sampleA2)
            sample = self.post_transform(sample)
            sampleA1 = self.post_transform(sampleA1)
            sampleA2 = self.post_transform(sampleA2)
            return sample['image'], sample['label'].unsqueeze(0), sampleA1['image'], sampleA2['image']
        else:
            sample = self.post_transform(sample)
        return sample['image'], sample['label'].unsqueeze(0)

    def __len__(self):
        return len(self.sample_list)

    def pre_transform(self):
        return Compose([
            RandomRotFlip(),
            RandomCrop(output_size=self.image_size)
        ])
    
    def aug_transform(self):
        return Compose([
            # RandomNoise(p=0.8),
            GaussianBlur(p=0.8)
        ])

    def post_transform(self):
        return Compose([
            CenterCrop(output_size=self.image_size),
            ToTensor()
        ])


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if torch.rand(1) < self.p:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            return {'image': image, 'label': label}
        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1, p=0.8):
        self.mu = mu
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if torch.rand(1) < self.p:
            noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
            noise = noise + self.mu
            image = image + noise
        return {'image': image, 'label': label}
    

class GaussianBlur(object):
    def __init__(self, sigma=(0.1, 2.), p=0.8):
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        if torch.rand(1) < self.p:
            image = gaussian_filter(image, sigma=sigma)
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}





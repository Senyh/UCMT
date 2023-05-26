import sys
import os
import random
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset, Subset, random_split
from copy import deepcopy
from data import transforms as T
from torchvision.transforms import *
import numpy as np
from PIL import Image
from skimage import io


class ISICDataset(Dataset):
    def __init__(self, image_path=None, stage='train', image_size=256, is_augmentation=False):
        super(ISICDataset, self).__init__()
        self.image_size = image_size
        self.sep = '\\' if sys.platform[:3] == 'win' else '/'
        self.stage = stage
        self.is_augmentation = is_augmentation
        if self.stage == 'train':
            self.image_files = sorted([os.path.join(image_path, x, y, z)
                                for x in os.listdir(image_path) if x.endswith('TrainDataset')
                                for y in os.listdir(os.path.join(image_path, x)) if y.endswith('images')
                                for z in os.listdir(os.path.join(image_path, x, y))])
            self.label_files = sorted([os.path.join(image_path, x, y, z)
                                for x in os.listdir(image_path) if x.endswith('TrainDataset')
                                for y in os.listdir(os.path.join(image_path, x)) if y.endswith('masks')
                                for z in os.listdir(os.path.join(image_path, x, y))])
        else:
            self.image_files = sorted([os.path.join(image_path, x, y, z)
                                for x in os.listdir(image_path) if x.endswith('TestDataset')
                                for y in os.listdir(os.path.join(image_path, x)) if y.endswith('images')
                                for z in os.listdir(os.path.join(image_path, x, y))])
            self.label_files = sorted([os.path.join(image_path, x, y, z)
                                for x in os.listdir(image_path) if x.endswith('TestDataset')
                                for y in os.listdir(os.path.join(image_path, x)) if y.endswith('masks')
                                for z in os.listdir(os.path.join(image_path, x, y))])
        assert len(self.image_files) == len(self.label_files)
        if self.is_augmentation:
            self.augmentation = self.augmentation_transform()
        self.post_transform = self.post_transform()
        self.label_transform = self.label_transform()
        self.pre_transform = self.pre_transform()

    def __getitem__(self, item):
        assert self.image_files[item].split(self.sep)[-1][:-4] == self.label_files[item].split(self.sep)[-1][:-4]
        image = io.imread(self.image_files[item]).astype('uint8')
        label = io.imread(self.label_files[item]).astype('bool').astype('uint8')  # 255 --> 1
        image = Image.fromarray(image).convert('RGB').resize((512, 512)) 
        label = Image.fromarray(label).convert('L').resize((512, 512)) 
        if self.stage == 'train':
            image, label = self.pre_transform(image, label)
            imageA1, imageA2 = deepcopy(image), deepcopy(image)
            imageA1, _ = self.augmentation(imageA1, label)
            imageA2, _ = self.augmentation(imageA1, label)
            image, label = self.post_transform(image), self.label_transform(label)
            imageA1 = self.post_transform(imageA1)
            imageA2 = self.post_transform(imageA2)
            label = torch.from_numpy(np.array(label)).unsqueeze(0)
            return image, label, imageA1, imageA2
        elif self.stage == 'test':
            image = self.post_transform(image)
            label = torch.from_numpy(np.array(label)).unsqueeze(0)
        else:
            image, label = self.post_transform(image), self.label_transform(label)
            label = torch.from_numpy(np.array(label)).unsqueeze(0)
        return image, label

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def augmentation_transform():
        return T.Compose([
            T.ColorJitter(0.5, 0.5, 0.5, 0.25),
            T.RandomPosterize(bits=5, p=0.2),
            T.RandomAutocontrast(p=0.2),
            T.RandomEqualize(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    def pre_transform(self):
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=180),
        ])

    def post_transform(self):
        return Compose([
            Resize([self.image_size, self.image_size], InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
        ])

    def label_transform(self):
        return Compose([
            Resize([self.image_size, self.image_size], InterpolationMode.NEAREST)
        ])
import random
from collections import defaultdict

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms 

from src.reproducibility import set_seed


class SSDADatasetWithLabel(Dataset):
    def __init__(self, imgs, labels, transform):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform:
            img = self.transform(img)
        return img, int(label)

    def __len__(self):
        return len(self.imgs)


class SSDADatasetWithoutLabel(Dataset):
    def __init__(self, imgs, weak_transform, strong_transform):
        self.imgs = imgs
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        img = self.imgs[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        weak_x = self.weak_transform(img)
        strong_x = self.strong_transform(img)
        return weak_x, strong_x

    def __len__(self):
        return len(self.imgs)


class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


class SrcDataset(Dataset):
    def __init__(self, dataset, weak_transform):
        self.dataset = dataset
        self.weak_transform = weak_transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        weak_x = self.weak_transform(x)
        return weak_x, y

    def __len__(self):
        return len(self.dataset)


class DstDataset(Dataset):
    def __init__(self, dataset, weak_transform, strong_transform):
        self.dataset = dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        weak_x = self.weak_transform(x)
        strong_x = self.strong_transform(x)
        return weak_x, strong_x

    def __len__(self):
        return len(self.dataset)


class CutoutAbs:
    def __init__(self, v):
        self.v = v

    def __call__(self, img):
        w, h = img.size
        x0 = np.random.uniform(0, w)
        y0 = np.random.uniform(0, h)
        x0 = int(max(0, x0 - self.v / 2.0))
        y0 = int(max(0, y0 - self.v / 2.0))
        x1 = int(min(w, x0 + self.v))
        y1 = int(min(h, y0 + self.v))
        xy = (x0, y0, x1, y1)
        # gray
        color = (127, 127, 127)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img


def get_train_weak_transform(
    img_size=32,
    mean=0.5,
    std=0.5,
):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.RandomCrop(size=img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )


def get_train_strong_transform(
    img_size=32,
    v=16,
    mean=0.5,
    std=0.5,
):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.RandAugment(magnitude=10),
            CutoutAbs(v),
            transforms.RandomCrop(size=img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )


def get_test_transform(
    img_size=32,
    mean=0.5,
    std=0.5,
):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )


def get_unlabeled_and_labeled_indices(
    labels: np.ndarray,
    num_labeled_indices_per_label,
    seed=None
):
    if seed is not None:
        set_seed(seed)
    indices_per_label = defaultdict(list)
    for i, label in enumerate(labels):
        indices_per_label[label].append(i)

    labled_indices = []
    for label in range(10):
        label_indices = indices_per_label[label]
        selected_label_indices = random.sample(
            label_indices,
            num_labeled_indices_per_label
        )
        labled_indices += selected_label_indices
    all_labels = set(range(len(labels)))
    unlabeled_indices = list(all_labels - set(labled_indices))
    return list(unlabeled_indices), list(labled_indices)

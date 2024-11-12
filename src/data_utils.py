import numpy as np
import PIL
from torch.utils.data import Dataset
from torchvision import transforms 


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

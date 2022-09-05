# -*- coding = utf-8 -*-
# @Time : 2022/5/1 10:33
# @Author : fan
# @File:data_aug.py
# @Software: PyCharm


from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

data_path = '/tmp/pycharm_project_382/data/'

test_transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

train_transforms = transforms.Compose([
    # transforms.Pad(padding=4),
    # transforms.CenterCrop(32),
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self):
        return self.dataset.__getitem__()

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(batch_size=128, root=data_path, train=True, shuffle=True, num_workers=4, n_items=-1):
    if train:
        dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=train_transforms)
    else:
        dataset = datasets.CIFAR10(root=root,train=train, download=True, transform=test_transforms)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader
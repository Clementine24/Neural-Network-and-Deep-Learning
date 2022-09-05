# -*- coding = utf-8 -*-
# @Time : 2022/4/23 16:16
# @Author : fan
# @File:utils.py
# @Software: PyCharm

import torch
import numpy as np
import random
from torch import nn
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

data_path = '/tmp/pycharm_project_382/data/'

def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

data_transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

def get_accuracy(model, dataloader, device='cpu'):
    correct = 0
    l = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(1)
            correct += torch.sum(pred == y).item()
            l += len(y)
    return correct / l

class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self):
        return self.dataset.__getitem__()

    def __len__(self):
        return min(self.n_items, len(self.dataset))

def get_cifar_loader(batch_size=128, root=data_path, train=True, shuffle=True, num_workers=4, n_items=-1):
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=data_transforms)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader

def init_weight(m):
    """
    针对ReLU激活函数的初始化方法，使用HE初始化方法或者正态初始化方法
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Conv2d') != -1:
        # nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.normal_(m.weight.data, mean = 0.0, std = 0.01)
    elif classname.find('Linear') != -1:
        # nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.normal_(m.weight.data, mean = 0.0, std = 0.01)
        nn.init.constant_(m.bias.data, 0.0)


def train(model, criterion, optimizer, train_loader, test_loader, epochs, device,
          scheduler=None, print_each=0, plot_test=False, title=None, save_model=False, save_path=None):
    model.to(device)
    train_accuracy_curve = [np.nan] * epochs
    test_accuracy_curve = [np.nan] * epochs
    losses_list = []

    for epoch in tqdm(range(epochs)):
        if scheduler is not None:
            scheduler.step()
        model.train()

        steps = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            losses_list.append(loss.item())
            optimizer.step()
            steps += 1

        if print_each and epoch % print_each == 0:
            print('Epoch {} training over, loss is {}'.format(epoch+1, loss.item()))

        model.eval()
        train_accuracy_curve[epoch] = get_accuracy(model, train_loader, device)
        test_accuracy_curve[epoch] = get_accuracy(model, test_loader, device)

    if save_model:
        torch.save(model.state_dict(), save_path)

    print('*******************************************************')
    print('After {} epochs, the accuracy on test set is {}'.format(epochs, test_accuracy_curve[-1]))

    # bin = 40
    x1 = range(0, epochs, 1)
    # x2 = range(0, len(losses_list), steps)
    # ax = plt.gca()
    # ax.patch.set_facecolor("#ECECF0")
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['bottom'].set_visible(False)  # 去掉下边框
    # ax.spines['left'].set_visible(False)  # 去掉左边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框
    # plt.grid(axis='both', color='w', linewidth=0.5)
    _, ax1 = plt.subplots()
    ax1.plot(x1, train_accuracy_curve, color='#77dfb1', label='train accuracy')
    if plot_test:
        ax1.plot(x1, test_accuracy_curve, color='#8ea4f7', label='test accuracy')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    # plt.legend(loc='upper right')
    plt.legend(loc='best')
    ax2 = ax1.twinx()
    ax2.plot(x1, losses_list[::steps], color='#fb3121', label='train loss')
    ax2.set_ylabel('loss')
    # plt.legend(loc='lower right')
    plt.legend(loc='best')
    plt.title(title)
    plt.show()

    return losses_list, train_accuracy_curve, test_accuracy_curve
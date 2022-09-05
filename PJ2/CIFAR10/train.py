# -*- coding = utf-8 -*-
# @Time : 2022/4/23 15:35
# @Author : fan
# @File:train.py
# @Software: PyCharm

import torch
from torch import optim
from tqdm import tqdm as tqdm
from utils import *
from model import *
import time

# 设置超参数
batch_size = 128
epoch = 20
lr = 1e-2
device = torch.device('cuda:0')
print(device)
print(torch.cuda.get_device_name(0))


if __name__ == '__main__':
    """
    基本网络结构部分探索代码
    """
    # 设置随机种子
    set_random_seeds(2022, device = device)

    # 数据加载和预处理
    train_loader = get_cifar_loader(batch_size=batch_size,train=True)
    test_loader = get_cifar_loader(batch_size=batch_size, train=False)

    # 定义神经网络
    model = ResNetC()
    # model.apply(init_weight) # 初始化参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

    # 开始训练
    start_time = time.time()
    _, _, _ = train(model, criterion, optimizer, train_loader, test_loader,
                    epoch, device, print_each=10, plot_test=True, title='ResNet with classifier')
    end_time = time.time()
    print('End of program operation! Total time: ', end_time - start_time)
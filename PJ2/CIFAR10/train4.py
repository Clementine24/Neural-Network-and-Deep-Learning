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
from lossfunc import *
import time
import data_aug
import cutmix
from torch.optim import lr_scheduler

# 设置超参数
batch_size = 128
# batch_size = 256
epoch = 165
# epoch = 154
# epoch = 20
# lr = 0.5
lr = 0.1
# lr = 0.05
T_0 = 205
T_mult = 1
device = torch.device('cuda:0')
model_path = '/tmp/pycharm_project_382/CIFAR10/modelsave/'
print(device)
print(torch.cuda.get_device_name(0))



if __name__ == '__main__':
    """
    其他模型增强实验部分探索代码
    """
    # 设置随机种子
    set_random_seeds(2022, device = device)

    # 数据加载和预处理
    train_loader = data_aug.get_cifar_loader(batch_size=batch_size,train=True)
    test_loader = data_aug.get_cifar_loader(batch_size=1024, train=False)


    # 定义神经网络
    model = ResNetC()
    # model = MyNet(drop_prob=0)
    # model = DenseNet(32, (6, 12, 24, 16), 64)
    model.apply(init_weight) # 初始化参数
    # criterion = smoothcrossentropy()
    # criterion = nn.CrossEntropyLoss()
    criterion = klloss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min=1e-3)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.2)

    # 开始训练
    start_time = time.time()
    # _, _, test_ac = cutmix.train(model, criterion, optimizer, train_loader, test_loader,
    #                 epoch, device, scheduler=scheduler, print_each=10, plot_test=True, title='GELU', save_model=True, save_path=model_path+'best.pth')
    cutmix.train(model, criterion, optimizer, train_loader, test_loader,
                 epoch, device, scheduler=scheduler, print_each=10, plot_test=True, title='GELU', save_model=True,
                 save_path=model_path + 'best.pth')
    end_time = time.time()
    print('End of program operation! Total time: ', end_time - start_time)
    # print('The last 20 test accuracy is:')
    # print(test_ac[-20:])



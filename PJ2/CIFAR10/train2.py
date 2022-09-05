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

# 设置超参数
batch_size = 128
epoch = 200
lr = 1e-2
device = torch.device('cuda:0')
print(device)
print(torch.cuda.get_device_name(0))


if __name__ == '__main__':
    """
    消融实验部分探索代码
    """
    # 设置随机种子
    set_random_seeds(2022, device = device)

    # 数据加载和预处理
    train_loader = get_cifar_loader(batch_size=batch_size,train=True)
    test_loader = get_cifar_loader(batch_size=batch_size, train=False)


    # 定义神经网络
    # model_lr_3 = ResNetC()
    model = ResNetC()
    # model = ResNet6conv()
    model.apply(init_weight) # 初始化参数
    # criterion = smoothcrossentropy()
    criterion = klloss()
    # criterion1 = smoothl1loss()
    # criterion2 = klloss()
    # criterion3 = smoothcrossentropy()
    # optimizer_lr = optim.SGD(model_lr_3.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    # weight_decay = [1e-4, 1e-3, 1e-2, 1e-1, 1, 5]
    #
    # 开始训练
    start_time = time.time()
    # _, _, _ = train(model_lr_3, criterion, optimizer_lr, train_loader, test_loader,
    #                                 epoch, device, print_each=10, plot_test=True, title='ResNet with classifier')
    _, _, _ = train(model, criterion, optimizer, train_loader, test_loader,
                                   epoch, device, print_each=10, plot_test=True, title='Res Net with smooth cross entropy')
    # model = ResNetC()
    # model.apply(init_weight)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # loss_l1, train_ac_l1, _ = train(model, criterion1, optimizer, train_loader, test_loader,
    #                 epoch, device, print_each=0, plot_test=True, title='Smooth L1 Loss function (with label smoothing)')
    # model = ResNetC()
    # model.apply(init_weight)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # loss_kl, train_ac_kl, _ = train(model, criterion2, optimizer, train_loader, test_loader,
    #                 epoch, device, print_each=0, plot_test=True, title='KL Loss function (with label smoothing)')
    # model = ResNetC()
    # model.apply(init_weight)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # loss_ce, train_ac_ce, _ = train(model, criterion3, optimizer, train_loader, test_loader,
    #                 epoch, device, print_each=0, plot_test=True, title='Smooth cross entropy Loss function (with label smoothing)')
    # for i in range(len(weight_decay)):
    #     weight = weight_decay[i]
    #     train_ac_list = np.zeros((len(weight_decay), epoch))
    #     model = ResNetC()
    #     model.apply(init_weight)
    #     optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=weight)
    #     _, train_ac_curve, _ = train(model, criterion, optimizer, train_loader, test_loader,
    #                                                    epoch, device, print_each=10, plot_test=True, title='weight decay = '+str(weight))
    #     train_ac_list[i] = train_ac_curve
    end_time = time.time()
    print('End of program operation! Total time: ', end_time - start_time)
    # x = range(len(loss_ce))
    # plt.plot(x, loss_ce, label='cross entropy')
    # plt.plot(x, loss_kl, label='kl divergence')
    # plt.plot(x, loss_l1, label='smooth l1')
    # plt.legend()
    # plt.xlabel('Step')
    # plt.ylabel('Loss')
    # plt.title('Loss on different loss function experiment')
    # plt.show()
    # x = range(len(train_ac_ce))
    # plt.plot(x, train_ac_ce, label='cross entropy')
    # plt.plot(x, train_ac_kl, label='kl divergence')
    # plt.plot(x, train_ac_l1, label='smooth l1')
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Train set accuracy on different loss function experiment')
    # plt.show()

    # x = range(epoch)
    # for i in range(len(weight_decay)):
    #     plt.plot(x, train_ac_list[i], label = 'weight decay='+str(weight_decay[i]))
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Train set accuracy on l2 regularization experiment')
    # plt.show()

    # x = range(len(train_ac_curve_0))
    # plt.plot(x, train_ac_curve_0, label='drop prob = 0 (No drop out)')
    # plt.plot(x, train_ac_curve_5, label='drop prob = 0.5')
    # plt.plot(x, train_ac_curve_8, label='drop prob = 0.8')
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Train set accuracy on Drop Out experience')
    # plt.show()
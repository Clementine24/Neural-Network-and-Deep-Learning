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
epoch = 40
lr = 1e-2
device = torch.device('cuda:0')
print(device)
print(torch.cuda.get_device_name(0))


if __name__ == '__main__':
    """
    优化实验部分探索代码
    """
    # 设置随机种子
    set_random_seeds(2022, device = device)

    # 数据加载和预处理
    train_loader = get_cifar_loader(batch_size=batch_size,train=True)
    test_loader = get_cifar_loader(batch_size=batch_size, train=False)


    # 定义神经网络
    model = ResNetC()
    model.apply(init_weight) # 初始化参数
    # criterion = smoothcrossentropy()
    criterion = klloss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))


    # 开始训练
    start_time = time.time()
    _, _, _ = train(model, criterion, optimizer, train_loader, test_loader,
                                   epoch, device, print_each=10, plot_test=True, title='Adam function')
    # model = ResNetC()
    # model.apply(init_weight)
    # optimizer_SGDM = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # loss_SGDM, train_ac_SGDM, test_ac_SGDM = train(model, criterion, optimizer_SGDM, train_loader, test_loader,
    #                 epoch, device, print_each=0, plot_test=True, title='SGDM optimizer')
    # model = ResNetC()
    # model.apply(init_weight)
    # optimizer_NAG = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # loss_NAG, train_ac_NAG, test_ac_NAG = train(model, criterion, optimizer_NAG, train_loader, test_loader,
    #                 epoch, device, print_each=0, plot_test=True, title='SGD with NAG optimizer')
    # model = ResNetC()
    # model.apply(init_weight)
    # optimizer_Adam = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    # loss_Adam, train_ac_Adam, test_ac_Adam = train(model, criterion, optimizer_Adam, train_loader, test_loader,
    #                 epoch, device, print_each=0, plot_test=True, title='Adam optimizer')
    # model = ResNetC()
    # model.apply(init_weight)
    # optimizer_AdamW = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=5e-4)
    # loss_AdamW, train_ac_AdamW, test_ac_AdamW = train(model, criterion, optimizer_AdamW, train_loader, test_loader,
    #                 epoch, device, print_each=0, plot_test=True, title='AdamW optimizer')
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

    # Plot the results
    # x = range(len(loss_SGDM))
    # plt.plot(x[::batch_size], loss_SGDM[::batch_size], label='SGDM')
    # plt.plot(x[::batch_size], loss_NAG[::batch_size], label='NAG')
    # plt.plot(x[::batch_size], loss_Adam[::batch_size], label='Adam')
    # plt.plot(x[::batch_size], loss_AdamW[::batch_size], label='AdamW')
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss on different optimizer experiment')
    # plt.show()
    # x = range(len(train_ac_SGDM))
    # plt.plot(x, train_ac_SGDM, label='SGDM')
    # plt.plot(x, train_ac_NAG, label='NAG')
    # plt.plot(x, train_ac_Adam, label='Adam')
    # plt.plot(x, train_ac_AdamW, label='AdamW')
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Train set accuracy on different optimizer experiment')
    # plt.show()
    # x = range(len(test_ac_SGDM))
    # plt.plot(x, test_ac_SGDM, label='SGDM')
    # plt.plot(x, test_ac_NAG, label='NAG')
    # plt.plot(x, test_ac_Adam, label='Adam')
    # plt.plot(x, test_ac_AdamW, label='AdamW')
    # plt.legend()
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Test set accuracy on different optimizer experiment')
    # plt.show()


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

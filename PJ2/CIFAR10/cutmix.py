# -*- coding = utf-8 -*-
# @Time : 2022/5/1 11:28
# @Author : fan
# @File:cutmix.py
# @Software: PyCharm


import numpy as np
from tqdm import tqdm
import torch
from utils import get_accuracy
import matplotlib.pyplot as plt

def rand_bbox(size, lam):
    """
    choose random cut mix area
    """
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 注意边界条件
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def train(model, criterion, optimizer, train_loader, test_loader, epochs, device,
          scheduler=None, print_each=0, plot_test=False, title=None, save_model=False, save_path=None,
          beta=1.0, cutmix_prob=1.0):
    """
    为cutmix单独定义一个训练函数
    """
    model.to(device)
    # train_accuracy_curve = [np.nan] * epochs
    # test_accuracy_curve = [np.nan] * epochs
    # losses_list = []
    # best_ac = 0.9508

    for epoch in tqdm(range(epochs)):
        if scheduler is not None:
            scheduler.step()
        model.train()

        steps = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            r = np.random.rand(1)
            if beta > 0 and r < cutmix_prob:
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(x.size(0)).to(device)
                target_a = y
                target_b = y[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                x[:,:,bbx1:bbx2,bby1:bby2] = x[rand_index,:,bbx1:bbx2,bby1:bby2]
                lam = 1 - ((bbx2-bbx1)*(bby2-bby1)/(x.size(-1)*x.size(-2)))
                out = model(x)
                loss = criterion(out, target_a) * lam + criterion(out, target_b) * (1. - lam)
            else:
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            # losses_list.append(loss.item())
            optimizer.step()
            steps += 1

        # model.eval()
        # train_accuracy_curve[epoch] = get_accuracy(model, train_loader, device)
        # test_accuracy_curve[epoch] = get_accuracy(model, test_loader, device)
        # if print_each and epoch % print_each == 0:
        #     print('Epoch {} training over, test accuracy is {}'.format(epoch+1, test_accuracy_curve[epoch]))


        # if save_model and test_accuracy_curve[epoch] >= best_ac:
        #     torch.save(model.state_dict(), save_path)
        #     print('At epoch {}, i update the best model, test ac is {}'.format(epoch+1, test_accuracy_curve[epoch]))
        #     best_ac = test_accuracy_curve[epoch]

    # print('*******************************************************')
    # print('After {} epochs, the accuracy on test set is {}'.format(epochs, test_accuracy_curve[-1]))

    # bin = 40
    # x1 = range(0, epochs, 1)
    # x2 = range(0, len(losses_list), steps)
    # ax = plt.gca()
    # ax.patch.set_facecolor("#ECECF0")
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['bottom'].set_visible(False)  # 去掉下边框
    # ax.spines['left'].set_visible(False)  # 去掉左边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框
    # plt.grid(axis='both', color='w', linewidth=0.5)
    # _, ax1 = plt.subplots()
    # ax1.plot(x1, train_accuracy_curve, color='#77dfb1', label='train accuracy')
    # if plot_test:
    #     ax1.plot(x1, test_accuracy_curve, color='#8ea4f7', label='test accuracy')
    # ax1.set_xlabel('epoch')
    # ax1.set_ylabel('accuracy')
    # # plt.legend(loc='upper right')
    # plt.legend(loc='best')
    # ax2 = ax1.twinx()
    # ax2.plot(x1, losses_list[::steps], color='#fb3121', label='train loss')
    # ax2.set_ylabel('loss')
    # plt.legend(loc='lower right')
    # # plt.legend(loc='best')
    # plt.title(title)
    # plt.show()

    return
    # return losses_list, train_accuracy_curve, test_accuracy_curve
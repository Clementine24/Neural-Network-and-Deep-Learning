# -*- coding = utf-8 -*-
# @Time : 2022/4/22 10:23
# @Author : fan
# @File:VGG_plot.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm as tqdm

lrs = [1e-3, 2e-3, 1e-4, 5e-4]

# 导入存储的训练数据
save_path = 'VGG_BN_landscape/'
vgg_train_ac = np.load(save_path+'vgg_train_ac.npy')
vgg_val_ac = np.load(save_path+'vgg_val_ac.npy')
vgg_loss = np.load(save_path+'vgg_loss.npy')
vgg_grad = np.load(save_path+'vgg_grad.npy')
vgg_weight = np.load(save_path+'vgg_weight.npy')

vgg_bn_train_ac = np.load(save_path+'vgg_bn_train_ac.npy')
vgg_bn_val_ac = np.load(save_path+'vgg_bn_val_ac.npy')
vgg_bn_loss = np.load(save_path+'vgg_bn_loss.npy')
vgg_bn_grad = np.load(save_path+'vgg_bn_grad.npy')
vgg_bn_weight = np.load(save_path+'vgg_bn_weight.npy')

# 可视化
fig_save = 'VGG_BN_landscape/plot_result/'
# 可视化VGG和VGG with BN的训练结果
def VGG_tra_res():
    ax = plt.gca()
    ax.patch.set_facecolor("#ECECF0")
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['bottom'].set_visible(False)  # 去掉下边框
    ax.spines['left'].set_visible(False)  # 去掉左边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.grid(axis='both', color='w', linewidth=0.5)
    x = range(1, vgg_train_ac.shape[1]+1, 1)
    plt.plot(x, vgg_train_ac[0], color='#57A86A', label = 'Standard VGG')
    plt.plot(x, vgg_bn_train_ac[0], color='#C44E52', label = 'Standard VGG + BatchNorm')
    plt.title('Training results with learning rate:'+str(lrs[0]))
    plt.xlabel('Epoch')
    plt.ylabel('Train set accuracy')
    plt.legend(loc='lower right')
    plt.show()


# 可视化loss landscape
def VGG_loss_land():
    bin = 20
    max_loss_land = np.max(vgg_loss, axis=0)[::bin]
    min_loss_land = np.min(vgg_loss, axis=0)[::bin]
    max_bn_loss_land = np.max(vgg_bn_loss, axis=0)[::bin]
    min_bn_loss_land = np.min(vgg_bn_loss, axis=0)[::bin]

    x = range(1, vgg_loss.shape[1], bin)
    ax = plt.gca()
    ax.patch.set_facecolor("#ECECF0")
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['bottom'].set_visible(False)  # 去掉下边框
    ax.spines['left'].set_visible(False)  # 去掉左边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.grid(axis='both', color='w', linewidth=0.5)
    plt.plot(x, max_loss_land, color='#57A86A')
    plt.plot(x, min_loss_land, color='#57A86A')
    plt.fill_between(x, max_loss_land, min_loss_land, facecolor = '#9FC8AC', label='Standard VGG')
    plt.plot(x, max_bn_loss_land, color='#C44E52')
    plt.plot(x, min_bn_loss_land, color='#C44E52')
    plt.fill_between(x, max_bn_loss_land, min_bn_loss_land, facecolor = '#DDA2A5', label='Standard VGG + BatchNorm')
    plt.legend(loc='upper right')
    plt.title('Loss Landscape')
    plt.xlabel('Steps')
    plt.ylabel('Loss Landscape')
    plt.show()

num_lr = len(lrs)
num_step = vgg_loss.shape[1]

# 可视化gradient predictiveness
def VGG_Grad_Pred():
    bin = 40
    grad_dis = np.zeros((num_lr, int(num_step-1)))
    for lr in range(num_lr):
        for step in range(num_step-1):
            grad_dis[lr][step] = np.linalg.norm(vgg_grad[lr*num_step+step]-vgg_grad[lr*num_step+step+1])
    max_grad_pred = np.max(grad_dis, axis=0)[::bin]
    min_grad_pred = np.min(grad_dis, axis=0)[::bin]

    grad_bn_dis = np.zeros((num_lr, num_step - 1))
    for lr in range(num_lr):
        for step in range(num_step - 1):
            grad_bn_dis[lr][step] = np.linalg.norm(vgg_bn_grad[lr*num_step+step] - vgg_bn_grad[lr*num_step+step + 1])
    max_bn_grad_pred = np.max(grad_bn_dis, axis=0)[::bin]
    min_bn_grad_pred = np.min(grad_bn_dis, axis=0)[::bin]

    x = range(1, num_step-1, bin)
    begin = 1
    x = x[begin:]
    max_grad_pred = max_grad_pred[begin:]
    min_grad_pred = min_grad_pred[begin:]
    max_bn_grad_pred = max_bn_grad_pred[begin:]
    min_bn_grad_pred = min_bn_grad_pred[begin:]
    ax = plt.gca()
    ax.patch.set_facecolor("#ECECF0")
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['bottom'].set_visible(False)  # 去掉下边框
    ax.spines['left'].set_visible(False)  # 去掉左边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.grid(axis='both', color='w', linewidth=0.5)
    plt.plot(x, max_grad_pred, color='#9FC8AC')
    plt.plot(x, min_grad_pred, color='#9FC8AC')
    plt.fill_between(x, max_grad_pred, min_grad_pred, facecolor = '#57A86A', label='Standard VGG')
    plt.plot(x, max_bn_grad_pred, color='#DDA2A5')
    plt.plot(x, min_bn_grad_pred, color='#DDA2A5')
    plt.fill_between(x, max_bn_grad_pred, min_bn_grad_pred, facecolor = '#C44E52', label='Standard VGG + BatchNorm')
    plt.legend(loc='upper right')
    plt.title('Gradient Predictiveness')
    plt.xlabel('Steps')
    plt.ylabel('Gradient Predictiveness')
    plt.show()


# 可视化beta_smoothness
def VGG_Beta_Smooth():
    bin = 20
    beta = np.zeros((num_lr, int(num_step-1)))
    for lr in range(num_lr):
        for step in range(num_step-1):
            # beta[lr][step] = np.linalg.norm(vgg_grad[lr*num_step+step]-vgg_grad[lr*num_step+step+1])/ \
            #                  (np.linalg.norm(vgg_grad[lr*num_step+step]*lrs[lr]) + 1e-3)
            beta[lr][step] = np.linalg.norm(vgg_grad[lr * num_step + step] - vgg_grad[lr * num_step + step + 1]) / \
                             np.linalg.norm(vgg_weight[lr * num_step + step] - vgg_weight[lr * num_step + step + 1])
    max_beta_smooth = np.max(beta, axis=0)[::bin]
    min_beta_smooth = np.min(beta, axis=0)[::bin]

    beta_bn = np.zeros((num_lr, num_step - 1))
    for lr in range(num_lr):
        for step in range(num_step - 1):
            # beta_bn[lr][step] = np.linalg.norm(vgg_bn_grad[lr*num_step+step] - vgg_bn_grad[lr*num_step+step + 1])/ \
            #                     (np.linalg.norm(vgg_bn_grad[lr*num_step+step]*lrs[lr])+1e-3)
            beta_bn[lr][step] = np.linalg.norm(vgg_bn_grad[lr * num_step + step] - vgg_bn_grad[lr * num_step + step + 1]) / \
                             np.linalg.norm(vgg_bn_weight[lr * num_step + step] - vgg_bn_weight[lr * num_step + step + 1])
    max_bn_beta_smooth = np.max(beta_bn, axis=0)[::bin]
    min_bn_beta_smooth = np.min(beta_bn, axis=0)[::bin]

    x = range(1, num_step-1, bin)
    begin = 10
    x = x[begin:]
    max_beta_smooth = max_beta_smooth[begin:]
    min_beta_smooth = min_beta_smooth[begin:]
    max_bn_beta_smooth = max_bn_beta_smooth[begin:]
    min_bn_beta_smooth = min_bn_beta_smooth[begin:]
    ax = plt.gca()
    ax.patch.set_facecolor("#ECECF0")
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['bottom'].set_visible(False)  # 去掉下边框
    ax.spines['left'].set_visible(False)  # 去掉左边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.grid(axis='both', color='w', linewidth=0.5)
    plt.plot(x, max_beta_smooth, color='#57A86A')
    plt.plot(x, min_beta_smooth, color='#57A86A')
    plt.fill_between(x, max_beta_smooth, min_beta_smooth, facecolor = '#9FC8AC', label='Standard VGG')
    plt.plot(x, max_bn_beta_smooth, color='#C44E52')
    plt.plot(x, min_bn_beta_smooth, color='#C44E52')
    plt.fill_between(x, max_bn_beta_smooth, min_bn_beta_smooth, facecolor = '#DDA2A5', label='Standard VGG + BatchNorm')
    plt.legend(loc='upper right')
    plt.title('effective beta-smoothness')
    plt.xlabel('Steps')
    plt.ylabel('beta-smoothness')
    plt.show()


if __name__=='__main__':
    print('Begin plot...')
    VGG_tra_res()
    print('Plot train results over!')
    # VGG_loss_land()
    print('Plot loss landscape over!')
    # VGG_Grad_Pred()
    print('Plot grad pred over!')
    # VGG_Beta_Smooth()
    print('All plot works end!')
# -*- coding = utf-8 -*-
# @Time : 2022/6/14 23:27
# @Author : fan
# @File:plot.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

log = np.load('../pretrain_checkpoint/logs.npy')
epoch_loss_list, acc_train_list, acc_test_list = log
plt.figure(figsize=(16,8))
plt.subplot(131)
plt.plot(range(1, 31), epoch_loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.subplot(132)
plt.plot(range(1, 31), acc_train_list)
plt.xlabel('epoch')
plt.ylabel('train accuracy')
plt.subplot(133)
plt.plot(range(1, 31), acc_test_list)
plt.xlabel('epoch')
plt.ylabel('test accuracy')
plt.show()
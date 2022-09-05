# -*- coding = utf-8 -*-
# @Time : 2022/5/1 23:34
# @Author : fan
# @File:train_lenet.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from slbi_toolbox import SLBI_ToolBox
from slbi_toolbox_adam import SLBI_ToolBox
from lenet import LeNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# 设置超参数
batch_size = 128
epochs = 20
lr = 1e-2
kappa = 1
interval = 20
mu = 20
device = torch.device('cuda:0')
print(device)
print(torch.cuda.get_device_name(0))
data_root = '/tmp/pycharm_project_382/DessiLBI/data/'
model_root = '/tmp/pycharm_project_382/DessiLBI/modelsave/'

# define some util function
def get_accuracy(model, dataloader, device='cpu'):
    model.eval()
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

def descent_lr(learning_rate, epoch, optimizer, interval):
        learning_rate = learning_rate * (0.1 ** (epoch //interval))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

def save_model_and_optimizer(model, optimizer, path):
    save_dict = {'model': model.state_dict(), 'optimizer':optimizer.state_dict()}
    torch.save(save_dict, path)

# load the data
transform = transforms.Compose([transforms.ToTensor(),])
train_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# define the model
model = LeNet().to(device)
name_list = []
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    print(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)

optimizer = SLBI_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, weight_decay=0)
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)

#train the model
train_accuracy_curve = [np.nan] * epochs
test_accuracy_curve = [np.nan] * epochs
loss_list = [np.nan] * epochs
for epoch in tqdm(range(epochs)):
    descent_lr(lr, epoch, optimizer, interval)
    loss_val = 0
    num = 0
    for iter, pack in enumerate(train_loader):
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        loss = F.nll_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = logits.max(1)
        loss_val += loss.item()
        num += data.shape[0]

    # loss_val /= num
    # train_ac = get_accuracy(model, train_loader, device=device)
    # test_ac = get_accuracy(model, test_loader, device=device)
    if (epoch + 1) % 1 == 0:
        print('*******************************')
        print('epoch : ', epoch + 1)
        print('loss : ', loss_val)
        # print('Accuracy : ', test_ac)
    # train_accuracy_curve[epoch] = train_ac
    # test_accuracy_curve[epoch] = test_ac
    loss_list[epoch] = loss_val

    optimizer.update_prune_order(epoch)

save_model_and_optimizer(model, optimizer, model_root+'train_lenet.pth')

# plot the result
x = range(epochs)
# plt.plot(x, train_accuracy_curve, label = 'train accuracy')
# plt.plot(x, test_accuracy_curve, label = 'test accuracy')
plt.plot(x, loss_list, label = 'loss')
plt.xlabel('epoch')
# plt.ylabel('accuracy')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()



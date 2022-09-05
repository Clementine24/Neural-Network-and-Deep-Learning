# -*- coding = utf-8 -*-
# @Time : 2022/5/2 15:05
# @Author : fan
# @File:train_cifar.py
# @Software: PyCharm

import torch
import numpy as np
import random
from vgg import VGG_A_BatchNorm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from slbi_toolbox_adam import SLBI_ToolBox
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置超参数
batch_size = 128
epochs = 20
lr = 1e-3
kappa = 1
interval = 20
mu = 20
device = torch.device('cuda:0')
print(device)
print(torch.cuda.get_device_name(0))
data_root = '/tmp/pycharm_project_382/data/'
model_root = '/tmp/pycharm_project_382/DessiLBI/modelsave/'

# define some util function
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

set_random_seeds(2022, device = device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# define the model
model = VGG_A_BatchNorm().to(device)
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
for epoch in tqdm(range(epochs)):
    model.train()
    descent_lr(lr, epoch, optimizer, interval)
    loss_val = 0
    num = 0
    for iter, pack in enumerate(train_loader):
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = logits.max(1)
        loss_val += loss.item()
        num += data.shape[0]

    loss_val /= num
    train_ac = get_accuracy(model, train_loader, device=device)
    test_ac = get_accuracy(model, test_loader, device=device)
    if (epoch + 1) % 1 == 0:
        print('*******************************')
        print('epoch : ', epoch + 1)
        print('loss : ', loss_val)
        print('Accuracy : ', test_ac)
    train_accuracy_curve[epoch] = train_ac
    test_accuracy_curve[epoch] = test_ac

    optimizer.update_prune_order(epoch)

save_model_and_optimizer(model, optimizer, model_root+'train_vgg.pth')

# plot the result
x = range(len(train_accuracy_curve))
plt.plot(x, train_accuracy_curve, label = 'train accuracy')
plt.plot(x, test_accuracy_curve, label = 'test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.title('DessiLBI with adam on CIFAR10')
plt.show()



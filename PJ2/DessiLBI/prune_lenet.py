# -*- coding = utf-8 -*-
# @Time : 2022/5/2 10:29
# @Author : fan
# @File:prune_lenet.py
# @Software: PyCharm


# from slbi_toolbox import SLBI_ToolBox
from slbi_toolbox_adam import SLBI_ToolBox
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lenet import LeNet

# 设置超参数
batch_size = 128
epochs = 20
lr = 1e-2
kappa = 1
interval = 20
mu = 20
device = torch.device('cuda:0')
M = 10
N = 10
print(device)
print(torch.cuda.get_device_name(0))
data_root = '/tmp/pycharm_project_382/DessiLBI/data/'
model_root = '/tmp/pycharm_project_382/DessiLBI/modelsave/'


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

torch.backends.cudnn.benchmark = True
load_pth = torch.load(model_root+'train_lenet.pth')
torch.cuda.empty_cache()
model = LeNet().to(device)
model.load_state_dict(load_pth['model'])
name_list = []
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)

transform = transforms.Compose([transforms.ToTensor(),])
test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


optimizer = SLBI_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, weight_decay=0)
optimizer.load_state_dict(load_pth['optimizer'])
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)

#### test prune one layer
print('prune conv3')
print('acc before pruning:',get_accuracy(model, test_loader, device))
weight_conv3 = model.conv3.weight.clone().detach().cpu().numpy()
before_weight = np.zeros((M*5,N*5))
for i in range(M):
    for j in range(N):
        before_weight[i*5:i*5+5, j*5:j*5+5] = weight_conv3[i][j]
before_weight = np.abs(before_weight)
optimizer.prune_layer_by_order_by_name(80, 'conv3.weight', True)
print('acc after pruning',get_accuracy(model, test_loader, device))
weight_conv3 = model.conv3.weight.clone().detach().cpu().numpy()
pruned_weight = np.zeros((M*5,N*5))
for i in range(M):
    for j in range(N):
        pruned_weight[i*5:i*5+5, j*5:j*5+5] = weight_conv3[i][j]
pruned_weight = np.abs(pruned_weight)
plt.subplot(121)
plt.imshow(before_weight, cmap='gray')
plt.axis('off')
plt.title('Conv3 before pruning')
plt.subplot(122)
plt.imshow(pruned_weight, cmap='gray')
plt.axis('off')
plt.title('Conv3 after pruning')
plt.show()


optimizer.recover()
print('acc after recovering',get_accuracy(model, test_loader, device))

#### test prune two layers

print('prune conv3 and fc1')
print('acc before pruning',get_accuracy(model, test_loader, device))
optimizer.prune_layer_by_order_by_list(80, ['conv3.weight', 'fc1.weight'], True)
print('acc after pruning',get_accuracy(model, test_loader, device))
optimizer.recover()
print('acc after recovering',get_accuracy(model, test_loader, device))




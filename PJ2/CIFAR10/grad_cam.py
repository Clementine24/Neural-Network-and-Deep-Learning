# -*- coding = utf-8 -*-
# @Time : 2022/5/3 12:49
# @Author : fan
# @File:grad_cam.py
# @Software: PyCharm

import numpy as np
import requests
import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import ResNetC

final_conv = 'conv5_x.convblock.3'
data_root = '/tmp/pycharm_project_382/data/'
model_path = '/tmp/pycharm_project_382/CIFAR10/modelsave/best.pth'
classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
batch_size = 1
M = 6
N = 3

# 图片预处理
def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img

fmap_block = []
grad_block = []
# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def forward_hook(module, input, output):
    fmap_block.append(output)

# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, y):
    H, W = img.shape[-2:]
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
    grads = grads.reshape([grads.shape[0],-1])
    weights = np.mean(grads, axis=1)
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # print(img.squeeze(0).size())
    cam_img = 0.3 * heatmap + 0.7 * img.squeeze(0).permute(1,2,0).numpy()

    plt.subplot(121)
    plt.imshow(img.squeeze(0).permute(1,2,0).numpy() * 0.5 + 0.2)
    plt.axis('off')
    plt.title('origin image: '+str(classes[y]))
    plt.subplot(122)
    plt.imshow(cam_img * 0.5 + 0.2)
    plt.axis('off')
    plt.title('Grad Cam')
    plt.show()

    # path_cam_img = os.path.join(out_dir, "cam.jpg")
    # cv2.imwrite(path_cam_img, cam_img)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# net.eval()

net = ResNetC()
net.load_state_dict(torch.load(model_path))

# 可视化卷积核
# plt.subplot(231)
# weight = net.conv1[0].weight.clone().detach().cpu().numpy()
# show_weight = np.zeros((N*3,M*3))
# for i in range(N):
#     for j in range(M):
#         show_weight[i*3:i*3+3, j*3:j*3+3] = weight[j][i]
# plt.imshow(show_weight)
# plt.axis('off')
# plt.title('the first conv layer')
# plt.subplot(232)
# weight = net.conv2_x[0].convblock[0].weight.clone().detach().cpu().numpy()
# show_weight = np.zeros((N*3,M*3))
# for i in range(N):
#     for j in range(M):
#         show_weight[i*3:i*3+3, j*3:j*3+3] = weight[j][i]
# plt.imshow(show_weight)
# plt.axis('off')
# plt.title('the second stage')
# plt.subplot(233)
# weight = net.conv3_x[0].convblock[0].weight.clone().detach().cpu().numpy()
# show_weight = np.zeros((N*3,M*3))
# for i in range(N):
#     for j in range(M):
#         show_weight[i*3:i*3+3, j*3:j*3+3] = weight[j][i]
# plt.imshow(show_weight)
# plt.axis('off')
# plt.title('the third stage')
# plt.subplot(234)
# weight = net.conv4_x[0].convblock[0].weight.clone().detach().cpu().numpy()
# show_weight = np.zeros((N*3,M*3))
# for i in range(N):
#     for j in range(M):
#         show_weight[i*3:i*3+3, j*3:j*3+3] = weight[j][i]
# plt.imshow(show_weight)
# plt.axis('off')
# plt.title('the fourth stage')
# plt.subplot(235)
# weight = net.conv5_x[0].convblock[0].weight.clone().detach().cpu().numpy()
# show_weight = np.zeros((N*3,M*3))
# for i in range(N):
#     for j in range(M):
#         show_weight[i*3:i*3+3, j*3:j*3+3] = weight[j][i]
# plt.imshow(show_weight)
# plt.axis('off')
# plt.title('the fifth stage')
# plt.show()


net.eval()
# print(net)

# net.conv1[0].register_forward_hook(forward_hook)
# net.conv1[0].register_backward_hook(backward_hook)
# net.conv2_x[0].convblock[0].register_forward_hook(forward_hook)
# net.conv2_x[0].convblock[0].register_backward_hook(backward_hook)
# net.conv3_x[0].convblock[0].register_forward_hook(forward_hook)
# net.conv3_x[0].convblock[0].register_backward_hook(backward_hook)
# net.conv4_x[0].convblock[3].register_forward_hook(forward_hook)
# net.conv4_x[0].convblock[3].register_backward_hook(backward_hook)
net.conv5_x[0].convblock[3].register_forward_hook(forward_hook)
net.conv5_x[0].convblock[3].register_backward_hook(backward_hook)


count = 0
for x, y in test_loader:
    if count != 16:
        count += 1
        continue
    output = net(x)
    max_idx = np.argmax(output.cpu().data.numpy())
    net.zero_grad()
    class_loss = output[0, max_idx]
    class_loss.backward()
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    # plt.subplot(153)
    # plt.imshow(x.numpy().squeeze()[0,:,:], cmap='gray')
    # plt.axis('off')
    # plt.title('Original image')
    # plt.show()
    # plt.subplot(151)
    # plt.imshow(fmap[0], cmap='gray')
    # plt.axis('off')
    # plt.subplot(152)
    # plt.imshow(fmap[1], cmap='gray')
    # plt.axis('off')
    # plt.subplot(153)
    # plt.imshow(fmap[2], cmap='gray')
    # plt.axis('off')
    # plt.subplot(154)
    # plt.imshow(fmap[3], cmap='gray')
    # plt.axis('off')
    # plt.subplot(155)
    # plt.imshow(fmap[4], cmap='gray')
    # plt.axis('off')
    # plt.show()
    cam_show_img(x, fmap, grads_val, y)
    # print(classes[y])
    # count += 1
    # if count == 10:
    break





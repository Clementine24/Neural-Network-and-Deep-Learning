# -*- coding = utf-8 -*-
# @Time : 2022/4/22 22:54
# @Author : fan
# @File:model.py
# @Software: PyCharm

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class ResBlock(nn.Module):
    """
    Basic block for ResNet
    """
    def __init__(self, in_channel, out_channel, is_bn=True, stride = 1):
        super(ResBlock, self).__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel) if is_bn else nn.Sequential(),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            # nn.ELU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel) if is_bn else nn.Sequential()
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x, is_plain=False):
        if not is_plain:
            block_out = self.convblock(x) + self.shortcut(x)
        else:
            block_out = self.convblock(x)
        return nn.ReLU(inplace=True)(block_out)
        # return nn.Tanh()(block_out)
        # return nn.ELU(inplace=True)(block_out)
        # return nn.LeakyReLU(inplace=True)(block_out)



class ResNet18(nn.Module):
    """
    ResNet-18 model
    """
    def __init__(self, num_class = 10, is_bn = True, is_drop = True):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64) if is_bn else nn.Sequential(),
            nn.ReLU(inplace=True)
        )
        self.conv2_x = nn.Sequential(ResBlock(64, 64, is_bn, 1), ResBlock(64, 64, is_bn, 1))
        self.conv3_x = nn.Sequential(ResBlock(64, 128, is_bn, 2), ResBlock(128, 128, is_bn, 1))
        self.conv4_x = nn.Sequential(ResBlock(128, 256, is_bn, 2), ResBlock(256, 256, is_bn, 1))
        self.conv5_x = nn.Sequential(ResBlock(256, 512, is_bn, 2), ResBlock(512, 512, is_bn, 1))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

class PlainBlock(nn.Module):
    """
    Basic block for PlainNet
    """
    def __init__(self, in_channel, out_channel, is_bn=True, stride = 1):
        super(PlainBlock, self).__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel) if is_bn else nn.Sequential(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel) if is_bn else nn.Sequential()
        )

    def forward(self, x):
        block_out = self.convblock(x)
        return nn.ReLU(inplace=True)(block_out)

class PlainNet18(nn.Module):
    """
    ResNet-18 model
    """
    def __init__(self, num_class = 10, is_bn = True, is_drop = True):
        super(PlainNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64) if is_bn else nn.Sequential(),
            nn.ReLU(inplace=True)
        )
        self.conv2_x = nn.Sequential(PlainBlock(64, 64, is_bn, 1), PlainBlock(64, 64, is_bn, 1))
        self.conv3_x = nn.Sequential(PlainBlock(64, 128, is_bn, 2), PlainBlock(128, 128, is_bn, 1))
        self.conv4_x = nn.Sequential(PlainBlock(128, 256, is_bn, 2), PlainBlock(256, 256, is_bn, 1))
        self.conv5_x = nn.Sequential(PlainBlock(256, 512, is_bn, 2), PlainBlock(512, 512, is_bn, 1))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

class ResNetC(nn.Module):
    """
    ResNet-18 with classifier model
    """
    def __init__(self, num_class = 10, is_bn = True, drop_prob=0.5):
        super(ResNetC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64) if is_bn else nn.Sequential(),
            nn.ReLU(inplace=True)
            # nn.Tanh()
            # nn.ELU(inplace=True)
            # nn.LeakyReLU(inplace=True)
        )
        self.conv2_x = nn.Sequential(ResBlock(64, 64, is_bn, 1), ResBlock(64, 64, is_bn, 1))
        self.conv3_x = nn.Sequential(ResBlock(64, 128, is_bn, 2), ResBlock(128, 128, is_bn, 1))
        self.conv4_x = nn.Sequential(ResBlock(128, 256, is_bn, 2), ResBlock(256, 256, is_bn, 1))
        self.conv5_x = nn.Sequential(ResBlock(256, 512, is_bn, 2), ResBlock(512, 512, is_bn, 1))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            # nn.Linear(512, 256),
            nn.GELU(),
            # nn.Tanh(),
            # nn.ELU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(4096, 1024),
            # nn.Linear(256, 128),
            nn.GELU(),
            # nn.Tanh(),
            # nn.ELU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(1024, num_class)
            # nn.Linear(128, num_class)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

class ResNet4conv(nn.Module):
    """
    Reduce the convolution layer.
    Only preserve 4 convolution layer.
    """
    def __init__(self, num_class = 10, is_bn = True, drop_prob=0.5):
        super(ResNet4conv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64) if is_bn else nn.Sequential(),
            nn.ReLU(inplace=True)
        )
        self.conv2_x = nn.Sequential(ResBlock(64, 64, is_bn, 1), ResBlock(64, 64, is_bn, 1))
        self.conv3_x = nn.Sequential(ResBlock(64, 128, is_bn, 2), ResBlock(128, 128, is_bn, 1))
        self.conv4_x = nn.Sequential(ResBlock(128, 256, is_bn, 2), ResBlock(256, 256, is_bn, 1))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        # out = self.conv5_x(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


class ResNet6conv(nn.Module):
    """
    Increase the convolution layer.
    Only preserve 4 convolution layer.
    """
    def __init__(self, num_class = 10, is_bn = True, drop_prob=0.5):
        super(ResNet6conv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64) if is_bn else nn.Sequential(),
            nn.ReLU(inplace=True)
        )
        self.conv2_x = nn.Sequential(ResBlock(64, 64, is_bn, 1), ResBlock(64, 64, is_bn, 1))
        self.conv3_x = nn.Sequential(ResBlock(64, 128, is_bn, 2), ResBlock(128, 128, is_bn, 1))
        self.conv4_x = nn.Sequential(ResBlock(128, 256, is_bn, 2), ResBlock(256, 256, is_bn, 1))
        self.conv5_x = nn.Sequential(ResBlock(256, 256, is_bn, 1), ResBlock(256, 256, is_bn, 1))
        self.conv6_x = nn.Sequential(ResBlock(256, 512, is_bn, 2), ResBlock(512, 512, is_bn, 1))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.conv6_x(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

class PreActBlock(nn.Module):
    """
    Pre-activation block for ResNet
    """

    def __init__(self, in_channel, out_channel, stride=1):
        super(PreActBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class MyNet(nn.Module):
    """
    ResNet after adding classifier and pre-activation block
    """
    def __init__(self, num_class = 10, drop_prob = 0.5):
        super(MyNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = PreActBlock(64, 64, stride=1)
        self.layer2 = PreActBlock(64, 128, stride=2)
        self.layer3 = PreActBlock(128, 256, stride=2)
        self.layer4 = PreActBlock(256, 512, stride=2)
        self.classifier = nn.Sequential(
            # nn.Linear(512, 4096),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            # nn.Linear(4096, 1024),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            # nn.Linear(1024, num_class)
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, growth_rate * bn_size, kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(growth_rate * bn_size))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(growth_rate * bn_size, growth_rate, kernel_size=3, stride=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""

    def __init__(self, num_input_feature, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(2, stride=2))


class _DenseBlock(nn.Sequential):
    """
    DenseBlock
    """

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module("denselayer%d" % (i + 1,), layer)


class DenseNet(nn.Module):
    """
    DenseNet model
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(3, stride=2, padding=1))
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                transition = _Transition(num_features, int(num_features * compression_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.avg_pool2d(features, 7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


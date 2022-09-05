import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, wide_resnet101_2, vgg19_bn, googlenet


class Recognizer(nn.Module):
	def __init__(self, num_classes=200):
		super().__init__()
		self.cnn = resnet18(pretrained=True)
		self.cnn.fc = nn.Linear(self.cnn.fc.in_features, num_classes)
		
	def forward(self, inputs):
		outputs = self.cnn(inputs)
		return outputs


class Recognizer_no_pre(nn.Module):
	def __init__(self, num_classes=200):
		super().__init__()
		self.cnn = resnet18(pretrained=False)
		self.cnn.fc = nn.Linear(self.cnn.fc.in_features, num_classes)

	def forward(self, inputs):
		outputs = self.cnn(inputs)
		return outputs


class Recognizer_resnet_50(nn.Module):
	def __init__(self, num_classes=200):
		super().__init__()
		self.cnn = resnet50(pretrained=True)
		self.cnn.fc = nn.Linear(self.cnn.fc.in_features, num_classes)

	def forward(self, inputs):
		outputs = self.cnn(inputs)
		return outputs

class Recognizer_wide_resnet(nn.Module):
	def __init__(self, num_classes=200):
		super().__init__()
		self.cnn = wide_resnet101_2(pretrained=True)
		self.cnn.fc = nn.Linear(self.cnn.fc.in_features, num_classes)

	def forward(self, inputs):
		outputs = self.cnn(inputs)
		return outputs

class Recognizer_vgg19bn(nn.Module):
	def __init__(self, num_classes=200):
		super().__init__()
		self.cnn = vgg19_bn(pretrained=True)
		self.cnn.fc = nn.Linear(self.cnn.classifier[6].in_features, num_classes)

	def forward(self, inputs):
		outputs = self.cnn(inputs)
		return outputs

class Recognizer_googlenet(nn.Module):
	def __init__(self, num_classes=200):
		super().__init__()
		self.cnn = googlenet(pretrained=True)
		self.cnn.fc = nn.Linear(self.cnn.fc.in_features, num_classes)

	def forward(self, inputs):
		outputs = self.cnn(inputs)
		return outputs

	
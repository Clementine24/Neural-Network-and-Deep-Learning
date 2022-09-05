# -*- coding = utf-8 -*-
# @Time : 2022/6/12 21:52
# @Author : fan
# @File:pretrain.py
# @Software: PyCharm
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image
import os
from utils import set_random_seeds, save_logs
from time import time
from tqdm import tqdm
from recognizer import Recognizer_no_pre
from torch import nn, optim

checkpoint_every = 1
checkpoint_path = '/tmp/pycharm_project_23/pretrain_checkpoint'
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)


class HWDBDataset(Data.Dataset):
    def __init__(self, mode='train', label_set={}):
        super(HWDBDataset, self).__init__()
        assert mode in ['train', 'test']
        self.mode = mode
        self.dirpath = '/tmp/pycharm_project_23/HWDB_data/'+mode
        self.images = None
        self.labels = None
        self.num_items = 0
        self.max_label = 500
        self.label_set = label_set
        self.read_data()

    def read_data(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        images = []
        labels = []
        count_label = 0
        for label in os.listdir(self.dirpath):
            if self.mode == 'train':
                if count_label == self.max_label:
                    break
                self.label_set[label] = count_label
                count_label += 1
                print(count_label)
            elif self.mode == 'test':
                if label not in self.label_set:
                    continue
            image_path = os.path.join(self.dirpath, label)
            for image_name in os.listdir(image_path):
                image = transform(Image.open(os.path.join(image_path, image_name)))
                images.append(image)
                labels.append(self.label_set[label])

        self.images = torch.stack(images)
        self.labels = torch.tensor(labels)
        self.num_items = len(images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.num_items


def train(model, iter_train, iter_test, criterion, optimizer, num_epochs, device=torch.device("cpu"), schedule=False):
    """
    训练模型的一般流程，并保存训练过程中的训练集损失、训练集准确率、测试集准确率。
    """
    print(f"Training on {device}")
    epoch_loss_list = []
    acc_train_list = []
    acc_test_list = []
    training_logs = (epoch_loss_list, acc_train_list, acc_test_list)
    best_acc_test = 0
    if schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs+5, eta_min=0, last_epoch=-1)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_total_train = 0
        num_corr_train = 0
        start_time = time()
        for inputs, labels in tqdm(iter_train, postfix={"epoch": epoch + 1}):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / len(iter_train)
            labels_hat = outputs.argmax(dim=1)
            num_total_train += len(labels_hat)
            num_corr_train += labels_hat.eq(labels).sum().item()
        acc_train = num_corr_train / num_total_train
        if epoch % checkpoint_every == 0:
            acc_test = evaluate_accuracy(model, iter_test, device)
            if acc_test > best_acc_test:
                best_acc_test = acc_test
                torch.save({'model': model.state_dict(), 'epoch':epoch}, os.path.join(checkpoint_path,'best_model.pth'))
            acc_test_list.append(acc_test)
            torch.save({'model': model.state_dict(), 'epoch':epoch}, os.path.join(checkpoint_path,'trained_model.pth'))
        using_time = time() - start_time
        epoch_loss_list.append(epoch_loss)
        acc_train_list.append(acc_train)
        if epoch % checkpoint_every == 0:
            print(f"Epoch: {epoch + 1}\tloss: {epoch_loss:.2f}\tacc_train: {acc_train:4.1%}\tacc_test: {acc_test:4.1%}\ttime: {using_time:.1f}s")
        else:
            print(f"Epoch: {epoch + 1}\tloss: {epoch_loss:.2f}\tacc_train: {acc_train:4.1%}\ttime: {using_time:.1f}s")
        if schedule:
            scheduler.step()
    return training_logs, best_acc_test


def evaluate_accuracy(model, data_iter, device):
    """
    评估模型在数据集上的准确率。
    """
    model.eval()
    num_total = 0
    num_corr = 0
    with torch.no_grad():
        for inputs, labels in data_iter:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            labels_hat = outputs.argmax(axis=1)
            num_total += len(labels_hat)
            num_corr += labels_hat.eq(labels).sum().item()
    accuracy = num_corr / num_total
    return accuracy

set_random_seeds(0)

batch_size = 10
learning_rate = 1e-3
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = HWDBDataset(mode="train")
dataset_test = HWDBDataset(mode="test", label_set=dataset_train.label_set)
iter_train = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
iter_test = Data.DataLoader(dataset_test, batch_size=400, shuffle=False)

model = Recognizer_no_pre(num_classes=500).to(device)
# state_dict = torch.load(os.path.join(checkpoint_path,'trained_model.pth'))
# model.load_state_dict(state_dict['model'])
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
logs, best_ac = train(model, iter_train, iter_test, criterion, optimizer, num_epochs, device)
save_logs(logs, f"../pretrain_checkpoint/logs")

print('The best accuracy on test dataset is', best_ac)
print("Done!")


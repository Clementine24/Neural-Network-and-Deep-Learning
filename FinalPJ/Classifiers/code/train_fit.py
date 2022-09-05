# -*- coding = utf-8 -*-
# @Time : 2022/6/10 15:30
# @Author : fan
# @File:train_fit.py
# @Software: PyCharm

# -*- coding = utf-8 -*-
# @Time : 2022/6/9 21:28
# @Author : fan
# @File:train_DA.py
# @Software: PyCharm
from recognizer import Recognizer, Recognizer_resnet_50, Recognizer_wide_resnet
from utils import *
from dataset import OracleImageDataset
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn

set_random_seeds(0)
shots = [1, 3, 5]

batch_size = 10
lr = 5e-4
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for shot in shots:
    dataset_train = OracleImageDataset(shot=shot, mode="train", DA=True)
    dataset_test = OracleImageDataset(shot=shot, mode="test")
    iter_train = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    iter_test = Data.DataLoader(dataset_test, batch_size=400, shuffle=False)

    model = Recognizer().to(device)
    # model = Recognizer_resnet_50().to(device)
    # model = Recognizer_wide_resnet().to(device)

    # fit the latest layer
    for para in list(model.parameters())[:-1]:
        para.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    _ = train(model, iter_train, iter_test, criterion, optimizer, 15, device)

    # finetune all layers
    for para in list(model.parameters()):
        para.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    _ = train(model, iter_train, iter_test, criterion, optimizer, 15, device)
    # save_logs(logs, f"checkpoints/resnet50_{shot}_shot_DA.npy")

    print("Done!")


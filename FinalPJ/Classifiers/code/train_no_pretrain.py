# -*- coding = utf-8 -*-
# @Time : 2022/6/10 16:28
# @Author : fan
# @File:train_no_pretrain.py
# @Software: PyCharm
from recognizer import Recognizer_no_pre
from utils import *
from dataset import OracleImageDataset
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn

set_random_seeds(0)
shots = [1, 3, 5]

batch_size = 10
lr = 1e-3
num_epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for shot in shots:
    # dataset_train = OracleImageDataset(shot=shot, mode="train", DA=True) # add DA
    dataset_train = OracleImageDataset(shot=shot, mode="train")  # No DA
    dataset_test = OracleImageDataset(shot=shot, mode="test")
    iter_train = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    iter_test = Data.DataLoader(dataset_test, batch_size=400, shuffle=False)

    model = Recognizer_no_pre().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logs = train(model, iter_train, iter_test, criterion, optimizer, num_epochs, device)
    save_logs(logs, f"checkpoints/resnet18_no_pretrain_{shot}_shot.npy")

    print("Done!")

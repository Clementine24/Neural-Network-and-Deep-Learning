# -*- coding = utf-8 -*-
# @Time : 2022/6/9 21:28
# @Author : fan
# @File:train_trad_DA.py
# @Software: PyCharm
from recognizer import *
from utils import *
from dataset import OracleImageDataset
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn

set_random_seeds(0)
shots = [1, 3, 5]

batch_size = 10
lr = 1e-3 # 1e-4 for other networks, 1e-3 for HWDB pretrain model
num_epochs = 30 # 200 for test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for shot in shots:
    dataset_train = OracleImageDataset(shot=shot, mode="train")
    dataset_test = OracleImageDataset(shot=shot, mode="test")
    iter_train = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    iter_test = Data.DataLoader(dataset_test, batch_size=400, shuffle=False)

    # model = Recognizer().to(device)
    # model = Recognizer_resnet_50().to(device)
    # model = Recognizer_wide_resnet().to(device)
    # model = Recognizer_vgg19bn().to(device)
    # model = Recognizer_googlenet().to(device)
    # HWDB pretrain model
    model = Recognizer_no_pre(num_classes=500).to(device)
    state_dict = torch.load(os.path.join('../pretrain_checkpoint', 'trained_model.pth'))
    model.load_state_dict(state_dict['model'])
    model.cnn.fc = nn.Linear(model.cnn.fc.in_features, 200)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logs, best_acc = train(model, iter_train, iter_test, criterion, optimizer, num_epochs, device)
    # save_logs(logs, f"checkpoints/wideresnet_{shot}_shot_DA.npy")

    print("The best test accuracy is", best_acc)
    print("Done!")

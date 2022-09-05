from recognizer import Recognizer
from utils import *
from dataset import OracleImageDataset
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn

set_random_seeds(1234)
shot = 1

batch_size = 10
learning_rate = 5e-4
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = OracleImageDataset(orc_DA_path=f"../data/DA_data/DA_{shot}_shot/orc_bert", shot=shot, mode="train", trad_DA=True)
dataset_test = OracleImageDataset(shot=shot, mode="test")
iter_train = Data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
iter_test = Data.DataLoader(dataset_test, batch_size=400, shuffle=False)

model = Recognizer().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
logs = train(model, iter_train, iter_test, criterion, optimizer, num_epochs, device)
save_logs(logs, f"resnet18_{shot}_shot_DA.npy")

print("Done!")

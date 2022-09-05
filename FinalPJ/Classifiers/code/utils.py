import numpy as np
from time import time
from tqdm import tqdm
import torch
import random
import os


def set_random_seeds(seed=0):
	"""
	Set random seeds for all possible random variable generators.
	:param seed: The given random seed (default 0)
	:return: None
	"""
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)  					# 为Python设置随机种子
	np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
	torch.manual_seed(seed)  # 固定随机种子（CPU）
	if torch.cuda.is_available():  # 固定随机种子（GPU)
		torch.cuda.manual_seed(seed)  # 为当前GPU设置
		torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
	# torch.backends.cudnn.deterministic = True  # 固定网络结构


def to_big_strokes(stroke, max_len=300):
	"""
	Converts from stroke-3 to stroke-5 format and pads to given length.
	"""
	# (But does not insert special start token).
	
	result = np.zeros((max_len, 5), dtype=float)
	l = len(stroke)
	assert l <= max_len
	result[0:l, 0:2] = stroke[:, 0:2]
	result[0:l, 3] = stroke[:, 2]
	result[0:l, 2] = 1 - result[0:l, 3]
	result[l:, 4] = 1
	return result


def to_normal_strokes(big_stroke):
	"""
	Convert from stroke-5 or stroke-4 format (from sketch-rnn paper) back to stroke-3.
	"""
	l = 0
	stroke_len = big_stroke.shape[1]
	if stroke_len == 4:
		l = len(big_stroke)
	else:
		for i in range(len(big_stroke)):
			if big_stroke[i, 4] > 0:
				l = i
			break
		if l == 0:
			l = len(big_stroke)
	result = np.zeros((l, 3))
	result[:, 0:2] = big_stroke[0:l, 0:2]
	result[:, 2] = big_stroke[0:l, 3]
	return result


def get_char_and_index_map():
	path = "../data/oracle_fs/seq/char_to_idx.txt"
	with open(path, encoding="utf-8") as f:
		idx2char = f.readline()
	char2idx = {char: idx for idx, char in enumerate(idx2char)}
	return idx2char, char2idx


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
		acc_test = evaluate_accuracy(model, iter_test, device)
		if acc_test > best_acc_test:
			best_acc_test = acc_test
		using_time = time() - start_time
		epoch_loss_list.append(epoch_loss)
		acc_train_list.append(acc_train)
		acc_test_list.append(acc_test)
		print(f"Epoch: {epoch + 1}\tloss: {epoch_loss:.2f}\tacc_train: {acc_train:4.1%}\tacc_test: {acc_test:4.1%}\ttime: {using_time:.1f}s")
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


def save_logs(logs, file_name="training_log.npy"):
	"""
	保存训练信息。
	"""
	logs_mat = np.stack(logs)
	np.save(file_name, logs_mat)

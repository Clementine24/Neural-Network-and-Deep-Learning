#_*_coding=utf-8#
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image
import os
from utils import *


class OracleImageDataset(Data.TensorDataset):
	def __init__(self, shot=1, mode="train", orc_DA_path=None, trad_DA=False):
		super().__init__()
		assert shot in [1, 3, 5]
		assert mode in ["train", "test"]
		self.num_items = 0
		self.shot = shot
		self.mode = mode
		self.images = None
		self.labels = None
		self.read_data(orc_DA_path)
		self.trad_DA = trad_DA
		self.random_transform = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(15)
		])
		
	def read_data(self, orc_DA_path=None):
		ori_path = f"../data/oracle_fs/img/oracle_200_{self.shot}_shot/{self.mode}"
		idx2char, char2idx = get_char_and_index_map()

		transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		
		images = []
		labels = []
		for char in os.listdir(ori_path):
			if len(char) == 1:
				image_path = os.path.join(ori_path, char)
				for image_name in os.listdir(image_path):
					image = transform(Image.open(os.path.join(image_path, image_name)).convert("RGB"))
					label = char2idx[char.encode('utf-8', errors='surrogateescape').decode('utf-8')]
					images.append(image)
					labels.append(label)

		if orc_DA_path is not None:
			for img_file in tqdm(os.listdir(orc_DA_path), total=10 * 200 * self.shot, postfix="Transforming..."):
				image = transform(Image.open(os.path.join(orc_DA_path, img_file)).convert("RGB"))
				img_name = img_file[:-4]
				_, mask_prob, img_id = img_name.split("_")
				if float(mask_prob) >= 0.08:
					continue
				label = int(img_id) // self.shot
				images.append(image)
				labels.append(label)

		self.images = torch.stack(images)
		self.labels = torch.tensor(labels)
		self.num_items = len(images)

	def __getitem__(self, item):
		if self.trad_DA:
			return self.random_transform(self.images[item]), self.labels[item]
		else:
			return self.images[item], self.labels[item]
	
	def __len__(self):
		return self.num_items
					

			
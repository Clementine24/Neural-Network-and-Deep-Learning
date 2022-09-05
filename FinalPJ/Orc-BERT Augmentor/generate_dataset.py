import numpy as np
import os
import pickle as pkl
from tqdm import tqdm

data_paths = [
	# f'testOracleDataset/npz/oracle_source_seq.npz'
	f"../../nndlOracle/data/oracle_fs/seq/oracle_200_5_shot/{i}.npz" for i in range(200)
]

offsets = {'train': {}, 'valid': {}}
modes = ['train', 'test']
cls_name = "source"
save_dir = 'testOracleDataset/dat'
ff = open('testOracleDataset/memmap_sum.txt', 'w')

for i, path in tqdm(enumerate(data_paths), total=200):
	cls_name = str(i)
	data = np.load(path, encoding='latin1', allow_pickle=True)
	tmp_num = 0
	for mode in modes:
		tmp_data = []
		data_mode = data[mode]
		if mode == "test":
			mode = "valid"
		for sketch in data_mode:
			tmp_data.append(sketch)
		save_path = os.path.join(save_dir, '{}_{}.dat'.format(cls_name, mode))
		offsets[mode][cls_name] = []
		start = 0
		max_len = 0
		len_record = []
		for sketch in tmp_data:
			if len(sketch.shape) != 2 or sketch.shape[1] != 3:
				print(sketch)
				continue
			end = start + sketch.shape[0]
			len_record.append(sketch.shape[0])
			max_len = max(max_len, sketch.shape[0])
			offsets[mode][cls_name].append((start, end))
			start = end
		len_record = np.array(len_record)
		tmp_num += len(tmp_data)
		
		stack_data = np.concatenate(tmp_data, axis=0)
		tmp_memmap = np.memmap(save_path, dtype=np.int16, mode='write', shape=stack_data.shape)
		tmp_memmap[:] = stack_data[:]
		tmp_memmap.flush()
	
	ff.write(f"{save_dir}/{cls_name}.dat\t{tmp_num}\n")
	with open('testOracleDataset/offsets.npz', 'wb') as f:
		pkl.dump(offsets, f)

ff.close()

print("Done!")

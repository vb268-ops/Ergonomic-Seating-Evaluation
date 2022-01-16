#Vaibhav Bisht
#MS, Mechanical Engineering, Cornell University
#Creating custom dataset as 'tensor' type

from sklearn.datasets import make_classification
import numpy as np
import torch
import pandas as pd

features = np.loadtxt("./ergoleft.txt", dtype='int32', delimiter=',')
label = np.loadtxt("./ergoright.txt", dtype='int32', delimiter=',')

class custom_dataset:
	def __init__(self, data, targets):
		self.data = data
		self.targets = targets

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		current_sample = self.data[idx, :]
		current_target = self.targets[idx]
		current_sample = torch.tensor(current_sample, dtype=torch.float)
		current_target = torch.tensor(current_target, dtype=torch.long)
		#The float and long here are the type required by PyTorch (Q)
		return(current_sample, current_target)
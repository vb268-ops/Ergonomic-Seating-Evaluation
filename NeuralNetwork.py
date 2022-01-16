#Vaibhav Bisht
#MS, Mechanical Engineering, Cornell University
#Deep neural network for classification of ergonomics

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from customdataset import custom_dataset #Will import a dictionary of tensors for the purpose of training

#Online resource says: epochs=11 is optimal in most cases, batch sizea usually less than 32 are helpful

#Enable GPU if present
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('You are working on', device)

#Hyper parameters
input_size = 4 #Number of attributes in input features
hidden_size_1 = 3 #Number of neurons in first hidden layer
hidden_size_2 = 2 #Number of neurons in second hidden layer
num_classes = 2 #Number of output classes
num_epochs = 20 #Number of epochs
batch_size = 4 #Batch size for training
learning_rate = 0.001 #Learning rate for weight updation

#FINDING & SPLITTING DATASET
features = np.loadtxt("./ergoleft.txt", dtype='int32', delimiter=',')
label = np.loadtxt("./ergoright.txt", dtype='int32', delimiter=',')

dataset = custom_dataset(data=features, targets=label)
data_size = len(dataset)
print('The total size of dataset is', data_size)
train = np.uint32(0.8*data_size)
test = np.uint32(0.2*data_size)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train, test])

#DataLoader helps us take input and label in a pair-wise manner with the respective batch size. 
#Shuffle changes the entities of the batch after every iteration
#train_loader holds 'iteration' number of batches of data
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
test_loader= torch.utils.data.DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

#CREATE NETWORK
class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
		super(NeuralNet, self).__init__()
		#This depicts one hidden layer
		#.cuda() done to load model into GPU(here, cuda)/ CPU
		self.l1 = nn.Linear(input_size, hidden_size_1).cuda() #This takes the input to hidden layer-1
		self.relu = nn.ReLU().cuda() #This performs activation over the weighted sum and bias addition of first hidden layer
		self.l2 = nn.Linear(hidden_size_1, hidden_size_2).cuda() #This takes from hidden layer-1 to hidden layer-2
		self.relu = nn.ReLU().cuda() #This performs activation over the weighted sum and bias addition of second hidden layer
		self.l3 = nn.Linear(hidden_size_2, num_classes).cuda() #This is the output layer, we use cross entropy loss later so no softmax for this (Q).

	def forward(self, x):
		out = self.l1(x)
		out = self.relu(out)
		out = self.l2(out)
		out = self.relu(out)
		out = self.l3(out)
		return out

model = NeuralNet(input_size, hidden_size_1, hidden_size_2, num_classes)

#LOSS FUNCTION & OPTIMIZER
criterion = nn.CrossEntropyLoss() #This creates the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #This performs the optimization over the loss function

#TRAINING LOOP
#This is same as the number of iterations
n_total_steps = len(train_loader)
print('Total number of steps/iterations, i.e., number of forward and backward passes in one epoch', n_total_steps)

for epoch in range(num_epochs):
	for i, (x, y) in enumerate(train_loader):
		
		print(f'Epoch {1+epoch}')
		print(f'i {1+i}')
		#.to(device) done to load model into GPU(here, cuda)/ CPU
		x = x.to(device)
		y = y.to(device)

		#FORWARD
		outputs = model(x)
		loss = criterion(outputs, y)

		#BACKWARD
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1) % batch_size == 0:
			print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

#TESTING MODEL
with torch.no_grad():
	n_correct = 0
	n_samples = 0
	for x, y in test_loader:
		x = x.to(device)
		y = y.to(device)
		outputs = model(x)

		#value, index 
		_, predictions = torch.max(outputs, 1)
		n_samples += y.shape[0]
		n_correct += (predictions == y).sum().item()

	accuracy = 100 * n_correct / n_samples
	print('\n Accuracy of Model:', accuracy, '%')
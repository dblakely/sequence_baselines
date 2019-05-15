### main.py

import numpy as np
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from utils import Vocabulary, FastaDataset, collate

#use_cuda = not args.no_cuda and torch.cuda.is_available()
use_cuda = True
device = torch.device('cuda' if use_cuda else 'cpu')
bsz = 32

class BetterLSTM(nn.Module):
	# input_size = alphabet_size
	def __init__(self, input_size, embedding_size, hidden_size, output_size, 
			n_layers=1, bidir=False, embedding=None):

		super(BetterLSTM, self).__init__()

		self.input_size = input_size
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.num_dir = 2 if bidir else 1

		# whether to use pre-trained embeddings
		if embedding:
			self.embedding = embedding
		else:
			self.embedding = nn.Embedding(num_embeddings=input_size, 
				embedding_dim=embedding_size)

		self.lstm = nn.LSTM(input_size=embedding_size, 
			hidden_size=hidden_size, 
			num_layers=n_layers, 
			bidirectional=bidir)
		self.fully_connected = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1) 

	def forward(self, x):
		'''
		embedded = self.embed(input)
		lstm_out, self.hidden = self.lstm(embedded, self.hidden)
		out = self.fully_connected(lstm_out[-1])
		scores = self.softmax(out)
		return scores
		'''
		# assumes x ~ (sequence_len, batch)
		batch_size = x.shape[1]
		hidden = self.init_hidden(batch=batch_size)
		embedded = self.embedding(x)
		output, (h_final, c_final) = self.lstm(embedded, hidden)
		out = self.fully_connected(h_final[-1])
		scores = self.softmax(out)

		return out

	def init_hidden(self, batch):
		h0 = torch.zeros(self.n_layers * self.num_dir, 
			batch, self.hidden_size, device=device)
		c0 = torch.zeros(self.n_layers * self.num_dir, 
			batch, self.hidden_size, device=device)
		return h0, c0

def main():
	trainset = FastaDataset('./data/1.1.train.fasta')
	train_loader = data.DataLoader(trainset, 
		batch_size=bsz, 
		shuffle=True, 
		drop_last=True,
		collate_fn=collate)
	alphabet = trainset.get_vocab()
	testset = FastaDataset('./data/1.1.test.fasta', alphabet)
	test_loader = data.DataLoader(testset, 
		batch_size=1, 
		shuffle=True,
		drop_last=True)
	model = BetterLSTM(input_size=alphabet.size(), 
				embedding_size=32,
				hidden_size=64, 
				output_size=2,
				n_layers=2, 
				bidir=True, 
				embedding=None).to(device)
	loss_function = F.cross_entropy

	opt = optim.Adam(model.parameters(), lr=0.001)

	#net = NeuralNetClassifier(BetterLSTM)

	# for CV
	params = {
		'lr': [0.01, 0.02],
		'max_epochs': [10, 20],
		'module__input_size': [alphabet.size()], 
		'module__embedding_size': [32], 
		'module__hidden_size': [16, 32], 
		'module__output_size': [2],
		'module__n_layers': [1], 
		'module__bidir': [False], 
		'module__embedding': [None]
	}
	'''
	grid_search = GridSearchCV(net, params, cv=3, scoring='accuracy')
	X, Y = [], []
	for x, y in train_loader:
		x = x.view(-1, 1)
		y = torch.squeeze(y, dim=0)
		X.append(x)
		Y.append(y)
	grid_search.fit(X, Y)
	print(grid_search.best_score_, grid_search.best_params_)
	'''
	count = 0
	for x, y in train_loader:
		#x = pad_sequence(x, padding_value=0, batch_first=True)
		count += 1
		opt.zero_grad()
		#y = torch.squeeze(y)
		x, y = x.to(device), y.to(device)
		y_pred = model(x)

		loss = loss_function(y_pred, y)
		loss.backward()
		opt.step()
		exit()

	with torch.no_grad():
		num_correct = 0
		true_ys = []
		preds = []
		scores = []
		num_samples = len(test_loader)
		for x, y in test_loader:
			y = torch.squeeze(y)
			x, y = x.to(device), y.to(device)
			out = model(x)
			print("out.shape = ", out.shape)
			max_index = out.max(dim=1)[1]
			num_correct += (max_index == y).sum().item()
			#pos_score = out[0][1].item()
			#scores.append(pos_score)
			#y_pred = out.argmax(dim=-1)
			#preds.append(y_pred)
			#true_ys.append(y)
			#if y_pred == y: num_correct += 1
		accuracy = (num_correct / num_samples) * 100
		print("accuracy = ", accuracy)

if __name__ == '__main__':
	main()
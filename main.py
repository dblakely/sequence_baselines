### main.py

import numpy as np
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import Vocabulary, FastaDataset, collate

#use_cuda = not args.no_cuda and torch.cuda.is_available()
use_cuda = True
device = torch.device('cuda' if use_cuda else 'cpu')
bsz = 32
epochs = 10

class SeqLSTM(nn.Module):
	# input_size = alphabet_size
	def __init__(self, input_size, embedding_size, hidden_size, output_size, 
			n_layers=1, bidir=False, embedding=None):

		super(SeqLSTM, self).__init__()

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

	def forward(self, x, lengths):
		# assumes x ~ (sequence_len, batch)
		batch_size = x.shape[1]
		hidden = self.init_hidden(batch=batch_size)
		embedded = self.embedding(x)
		embedded = pack_padded_sequence(embedded, lengths)
		lstm_out, (h_final, c_final) = self.lstm(embedded, hidden)
		output = pad_packed_sequence(lstm_out)
		out = self.fully_connected(h_final[-1])
		scores = self.softmax(out)

		return out

	def init_hidden(self, batch):
		h0 = torch.zeros(self.n_layers * self.num_dir, 
			batch, self.hidden_size, device=device)
		c0 = torch.zeros(self.n_layers * self.num_dir, 
			batch, self.hidden_size, device=device)
		return h0, c0

def train_epoch(model, opt, train_loader):
	for x, y, lengths in train_loader:
		opt.zero_grad()
		x, y = x.to(device), y.to(device)
		y_pred = model(x, lengths)
		loss = F.cross_entropy(y_pred, y)
		loss.backward()
		opt.step()

def test_epoch(model, test_loader):
	with torch.no_grad():
		num_correct = 0
		true_ys = []
		preds = []
		scores = []
		num_samples = len(test_loader)

		for x, y, lengths in test_loader:
			x, y = x.to(device), y.to(device)
			out = model(x, lengths)
			y_pred = out.max(dim=1)[1]
			num_correct += (y_pred == y).sum().item()
			pos_score = out[0][1].item()

			true_ys += y.tolist()
			scores.append(pos_score)
			preds += y_pred.tolist()

	accuracy = (num_correct / num_samples) * 100
	confusion = metrics.confusion_matrix(true_ys, preds)
	print(str(confusion) + '\n')
	# true positive rate/sensitvity
	tpr = 100 * confusion[1][1] / (confusion[1][0] + confusion[1][1])
	# true negative rate/specificity
	tnr = 100 * confusion[0][0] / (confusion[0][0] + confusion[0][1])
	# AUROC
	auc = metrics.roc_auc_score(true_ys, scores)
	print("acc = {}, tpr/sensitvity = {},"
		"tnr/specificity = {}, AUROC = {}".format(accuracy, tpr, tnr, auc))
	return accuracy

def main():
	trainset = FastaDataset('./data/1.1.train.fasta')
	train_loader = data.DataLoader(trainset, 
		batch_size=bsz, 
		shuffle=True,
		collate_fn=collate)
	alphabet = trainset.get_vocab()
	testset = FastaDataset('./data/1.1.test.fasta', alphabet)
	test_loader = data.DataLoader(testset, 
		batch_size=1, 
		shuffle=True,
		collate_fn=collate)
	model = SeqLSTM(input_size=alphabet.size(), 
				embedding_size=32,
				hidden_size=64, 
				output_size=2,
				n_layers=2, 
				bidir=True, 
				embedding=None).to(device)

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

	for i in trange(1, epochs + 1):
		train_epoch(model, opt, train_loader)
		test_epoch(model, test_loader)

	accuracy = test_epoch(model, test_loader)
	print("accuracy = {}".format(accuracy))

if __name__ == '__main__':
	main()

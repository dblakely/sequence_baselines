### main.py

import numpy as np
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn import metrics
from tqdm import tqdm, trange
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import Vocabulary, FastaDataset, collate

def get_args():
	parser = argparse.ArgumentParser(description='Bio-Sequence RNN Baselines')
	parser.add_argument('-b', '--batch', type=int, default=64, metavar='N',
		help='input batch size for training (default: 64)')
	parser.add_argument('--trn', type=str, required=True, help='Training file', metavar='1.1.train.fasta')
	parser.add_argument('--tst', type=str, required=True, help='Test file', metavar='1.1.test.fasta')
	parser.add_argument('--file', type=str, required=True, help='File to gris search results to')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
	parser.add_argument('--num-folds', type=int, default=7, help='Number of folds for CV')

	return parser.parse_args()

args = get_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print("device = ", device)
bsz = args.batch
train_file = args.trn
test_file = args.tst
output_file = args.file
highest_auc = 0
best_params = {}
num_folds = args.num_folds

with open(output_file, 'w+') as f:
	f.write("trn: {} tst: {}, batch: {}, out: {}".format(train_file,
		test_file, bsz, output_file))

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
		packed_embedded = pack_padded_sequence(embedded, lengths)
		lstm_out, (h_final, c_final) = self.lstm(packed_embedded, hidden)
		lstm_out = pad_packed_sequence(lstm_out)
		logits = self.fully_connected(h_final[-1])

		return logits

	def init_hidden(self, batch):
		h0 = torch.zeros(self.n_layers * self.num_dir, 
			batch, self.hidden_size, device=device)
		c0 = torch.zeros(self.n_layers * self.num_dir, 
			batch, self.hidden_size, device=device)
		return h0, c0

def train_epoch(model, opt, train_loader):
	num_batches = len(train_loader)
	epoch_loss = 0
	for x, y, lengths in train_loader:
		opt.zero_grad()
		x, y = x.to(device), y.to(device)
		logits = model(x, lengths)
		loss = F.cross_entropy(logits, y)
		loss.backward()
		opt.step()
		epoch_loss += loss.item()
	
	return epoch_loss / num_batches

def evaluate(model, test_loader):
	num_batches = len(test_loader)
	num_samples = 0
	epoch_loss = 0
	num_correct = 0
	true_ys = []
	preds = []
	scores = []
	with torch.no_grad():
		for x, y, lengths in test_loader:
			num_samples += y.shape[0]
			x, y = x.to(device), y.to(device)
			logits = model(x, lengths)
			loss = F.cross_entropy(logits, y)
			epoch_loss += loss.item()
			y_pred = logits.max(dim=1)[1]
			num_correct += (y_pred == y).sum().item()
			probs = F.softmax(logits, dim=1)
			pos_scores = probs[:,1].tolist()
			scores += pos_scores
			pos_score = probs[0][1].item()
			true_ys += y.tolist()
			preds += y_pred.tolist()

	epoch_loss /= num_batches
	accuracy = (num_correct / num_samples) * 100
	confusion = metrics.confusion_matrix(true_ys, preds)
	#print(str(confusion) + '\n')
	# true positive rate/sensitvity
	tpr = 100 * confusion[1][1] / (confusion[1][0] + confusion[1][1])
	# true negative rate/specificity
	tnr = 100 * confusion[0][0] / (confusion[0][0] + confusion[0][1])
	# AUROC
	try:
		auc = metrics.roc_auc_score(true_ys, scores)
	except ValueError as e:
		with open(output_file, 'a+') as f:
			f.write(str(e) + "\n")
			f.write("true_ys = {}\n, scores = {}".format(true_ys, scores))
		auc = 0
	
	return epoch_loss, accuracy, tpr, tnr, auc

def run(params, trainset):
	global highest_auc, best_params
	print(params)

	trainset.split()
	total_acc = 0
	total_auc = 0

	for i in range(num_folds):
		train, vali = trainset.get_fold(i)
		train_loader = data.DataLoader(train, 
			batch_size=bsz, 
			shuffle=False,
			collate_fn=collate)
		vali_loader = data.DataLoader(vali,
			batch_size=bsz,
			shuffle=False,
			collate_fn=collate)
		model = SeqLSTM(input_size=params['input_size'],
			embedding_size=params['embedding_size'],
			hidden_size=params['hidden_size'],
			output_size=params['output_size'],
			n_layers=params['n_layers'],
			bidir=params['bidir'],
			embedding=None).to(device)
		opt = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=0.0005)

		for i in range(1, 21):
			train_loss = train_epoch(model, opt, train_loader)
			vali_loss, acc, tpr, tnr, auc = evaluate(model, vali_loader)
			# early stopping criterion
			if vali_loss > train_loss:
				break

		vali_loss, acc, tpr, tnr, auc = evaluate(model, vali_loader)
		result = "train_loss = {}, vali_loss = {}, ".format(train_loss, vali_loss)
		result += "acc = {}, tpr/sensitvity = {}, ".format(acc, tpr)
		result += "tnr/specificity = {}, AUROC = {}".format(tnr, auc)
		total_acc += acc
		total_auc += auc

	acc = total_acc / num_folds
	auc = total_auc / num_folds

	if (auc > highest_auc):
		highest_auc = auc
		best_params = params

	with open(output_file, 'a+') as f:
		f.write("\n\n" + str(params) + '\n' + result + '\n')

def run_best(trainset, testset):
	trainset.split()
	train, vali = trainset.get_fold(0)


	train_loader = data.DataLoader(train, 
		batch_size=bsz, 
		shuffle=False,
		collate_fn=collate)
	vali_loader = data.DataLoader(vali,
		batch_size=bsz,
		shuffle=False,
		collate_fn=collate)
	test_loader = data.DataLoader(testset,
		batch_size=bsz,
		shuffle=True,
		collate_fn=collate)
	model = SeqLSTM(input_size=best_params['input_size'],
		embedding_size=best_params['embedding_size'],
		hidden_size=best_params['hidden_size'],
		output_size=best_params['output_size'],
		n_layers=best_params['n_layers'],
		bidir=best_params['bidir'],
		embedding=None).to(device)
	opt = optim.Adam(model.parameters(), lr=best_params['lr'])

	for i in range(1, 21):
		train_loss = train_epoch(model, opt, train_loader)
		vali_loss, acc, tpr, tnr, auc = evaluate(model, vali_loader)
		if vali_loss > train_loss:
			break

	test_loss, test_acc, test_tpr, test_tnr, test_auc = evaluate(model, test_loader)
	result = "loss = {}, acc = {}, ".format(test_loss, test_acc)
	result += "tpr/sensitvity = {}, tnr/specificity = {}, ".format(test_tpr, test_tnr)
	result += "AUROC = {}".format(test_auc)

	with open(output_file, 'a+') as f:
		f.write("\n\nFinal model: " + str(best_params) + '\n' + result + '\n')

def main():
	trainset = FastaDataset(train_file)
	alphabet = trainset.get_vocab()
	testset = FastaDataset(test_file, alphabet)

	param_space = {
		'lr': [0.0001],
		'input_size': [alphabet.size()], 
		'embedding_size': [32, 64, 128, 256], 
		'hidden_size': [32, 64, 128, 256],
		'output_size': [2],
		'n_layers': [1, 2, 3, 4],
		'bidir': [True],
		'optimizer': ['Adam']
	}

	param_list = list(ParameterGrid(param_space))

	count = 0
	for params in param_list:
		count += 1
		run(params, trainset)
		if (count % 5 == 0):
			run_best(trainset, testset)

	run_best(trainset, testset)

if __name__ == '__main__':
	main()

### seq_rnn.py
### RNN baseline models for comparisons with iGakco-SVM
### Derrick Blakely, December 2018

### General Imports
import numpy as np
import os
import sys
import shutil
import matplotlib
# A less ad hoc way of setting these backends would be nice
if os.environ['HOME'] == "/Users/derrick":
	matplotlib.use('TkAgg') # Need to use TkAgg backend for my machine
else:
	matplotlib.use('Agg') # Need to use Agg backend for the qdata nodes
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import random
import argparse
from dataset import Dataset
from tqdm import tqdm, trange
from sklearn import metrics
import datetime
import seaborn as sn
import pandas as pd

### Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def get_args():
	parser = argparse.ArgumentParser(description='Bio-Sequence RNN Baselines')
	parser.add_argument('-b', '--batch', type=int, default=64, metavar='N',
		help='input batch size for training (default: 64)')
	parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
		help='number of epochs to train (default: 10)')
	parser.add_argument('-i', '--iters', type=int, default=1000, metavar='N',
		help='number of iterations to train (default: 1000)')
	parser.add_argument('-lr', '--learning-rate', type=float, default=0.01, metavar='LR',
		help='learning rate (default: 0.01)')
	parser.add_argument('-em', '--embed-size', type=int, default=32,
		help='Size of the embedding space (using char-level embeddings')
	parser.add_argument('--layers', type=int, default=1, metavar='N',
		help='Number of RNN layers to stack')
	parser.add_argument('--bidir', action='store_true', default=False,
		help='Whether to use a bidirectional RNN')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
		help='SGD momentum (default: 0.5)')
	parser.add_argument('--hidden', type=int, default=128, metavar='N',
		help='Number of hidden units (default: 128)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
		help='disables CUDA')
	parser.add_argument('-li', '--log-interval', type=int, default=1000, metavar='N',
		help='how many iterations to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=False,
		help='For Saving the current Model')
	parser.add_argument('--opt', choices=['adagrad', 'adam', 'sgd'], default='sgd',
		help='Which optimizers to use. Options are Adagrad, Adam, SGD')
	parser.add_argument('--trn', type=str, required=True, help='Training file', metavar='1.1.train.fasta')
	parser.add_argument('--tst', type=str, required=True, help='Test file', metavar='1.1.test.fasta')
	parser.add_argument('--show-graphs', action='store_true', default=False,
		help='Will show plots of the training and test accuracy and the training loss over time')
	parser.add_argument('-od', '--output-directory', type=str,
		help="""Name of directory to create. Will save data inside the directory.
				If not provided, logged data will not be saved.""")
	parser.add_argument('-d', '--dict', required=True, type=str, metavar='dna.dictionary',
		help='Dictionary file containing all chars that can appear in sequences, 1 per line')
	
	return parser.parse_args()

args = get_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

'''Change visible devices with:
$ CUDA_VISIBLE_DEVICES=0 python seq_rnn.py [args]
'''
device = torch.device('cuda' if use_cuda else 'cpu')
dict_file = args.dict
train_file = args.trn
test_file = args.tst
iters = args.iters
log_interval = args.log_interval
n_layers = args.layers
bidir = args.bidir
BATCH = args.batch
embed_size = args.embed_size
lr = args.learning_rate
PAD_VAL = -1

class History(object):
	def __init__(self):
		self.acc_iters = []
		self.train_acc = []
		self.test_acc = []
		self.loss_iters = []
		self.losses = []
		self.auc_iters = []
		self.train_auc = []
		self.test_auc = []
		self.fpr, self.tpr = [], []

	def add_acc(self, iter, train, test):
		self.acc_iters.append(iter)
		self.train_acc.append(train)
		self.test_acc.append(test)

	# training loss (e.g., NLL Loss)
	def add_loss(self, iter, loss):
		self.loss_iters.append(iter)
		self.losses.append(loss)

	# for plotting changes in auc-roc over time
	def add_auc(self, iter, train, test):
		self.auc_iters.append(iter)
		self.train_auc.append(train)
		self.test_auc.append(test)

	# data needed for plotting a single ROC
	def add_roc_info(self, fpr, tpr):
		self.fpr = fpr
		self.tpr = tpr

	def plot_acc(self, show=False, path=None):
		plt.plot(self.acc_iters, self.train_acc, label='Train Accuracy')
		plt.plot(self.acc_iters, self.test_acc, label='Test Accuracy')
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Training Iteration')
		plt.legend(['Train', 'Test'], loc='upper left')
		if show: plt.show()
		if path is not None:
			file_name = os.path.join(path, 'accuracy.pdf')
			plt.savefig(file_name)
			plt.clf()

	def plot_loss(self, show=False, path=None):
		plt.plot(self.loss_iters, self.losses, label='NLL Loss')
		plt.title('Model Training Loss')
		plt.ylabel('Negative Log-Likelihood Loss')
		plt.xlabel('Training Iteration')
		if show: plt.show()
		if path is not None:
			file_name = os.path.join(path, 'loss.pdf')
			plt.savefig(file_name)
			plt.clf()

	# test auc vs iters plot
	# shows changes in auc-roc over time (not an ROC curve itself)
	def plot_auc(self, show=False, path=None):
		plt.plot(self.auc_iters, self.test_auc, label='Test AUC')
		plt.title('Test Set AUC-ROC vs Training Iterations')
		plt.ylabel('AUC-ROC')
		plt.xlabel('Training Iteration')
		if show: plt.show()
		if path is not None:
			file_name = os.path.join(path, 'auc.pdf')
			plt.savefig(file_name)
			plt.clf()
	
	# an actual ROC
	def plot_roc(self, show=False, path=None):
		try:
			plt.plot(self.fpr, self.tpr, label='ROC Curve')
			plt.title('ROC Curve')
			plt.ylabel('TPR')
			plt.xlabel('FPR')
			if show: plt.show()
			if path is not None:
				file_name = os.path.join(path, 'roc.pdf')
				plt.savefig(file_name)
				plt.clf()
		except:
			print("Make sure the History add_roc_info() was called first!")

	def save_data(self, file_prefix):
		train_acc_file = file_prefix + ".train.acc"
		test_acc_file = file_prefix + ".test.acc"
		train_loss_file = file_prefix + ".train.loss"
		auc_vs_iters_file = file_prefix + '.test.auc'
		roc_file = file_prefix + '.roc'
		with open(train_acc_file, 'w+') as f:
			for i, acc in zip(self.acc_iters, self.train_acc):
				f.write("{} {}\n".format(i, acc))
		with open(test_acc_file, 'w+') as f:
			for i, acc in zip(self.acc_iters, self.test_acc):
				f.write("{} {}\n".format(i, acc))
		with open(train_loss_file, 'w+') as f:
			for i, loss in zip(self.loss_iters, self.losses):
				f.write("{} {}\n".format(i, loss))
		with open(auc_vs_iters_file, 'w+') as f:
			for i, auc in zip(self.auc_iters, self.test_auc):
				f.write("{} {}\n".format(i, auc))
		with open(roc_file, 'w+') as f:
			f.write(' '.join(map(str, self.fpr)))
			f.write('\n')
			f.write(' '.join(map(str, self.tpr)))

class Evaluation(object):
	def __init__(self, model, samples, labels):
		"""
		Arguments
		---------
		model: an LSTM
		samples: array of test or train tensors
		labels: array of label tensors
		"""

		num_samples = len(samples)
		assert num_samples == len(labels)
		
		num_correct = 0
		true_ys = []
		preds = []
		scores = []



		with torch.no_grad():
			for x, y in zip(samples, labels):
				x = x.unsqueeze(1)
				y = y.item()

				h0, c0 = model.init_hidden(batch=1)
				out = model(x, h0, c0)
				scores.append(torch.max(out).item())
				y_pred = out.argmax(dim=-1).item()
				preds.append(y_pred)
				true_ys.append(y)
				if y_pred == y: num_correct += 1

		self.accuracy = (num_correct / num_samples) * 100
		self.confusion = metrics.confusion_matrix(true_ys, preds)

		self.increasing_fprs, self.increasing_tprs, thresholds = metrics.roc_curve(true_ys, scores)
		self.auc = metrics.roc_auc_score(true_ys, scores)

		# true positive rate/sensitvity
		self.tpr = 100 * self.confusion[1][1] / (self.confusion[1][0] + self.confusion[1][1])
		# true negative rate/specificity
		self.tnr = 100 * self.confusion[0][0] / (self.confusion[0][0] + self.confusion[0][1])

	def show_confusion(self):
		print(str(self.confusion) + '\n')

	def plot_confusion(self, show=False, path=None):
		dataframe = pd.DataFrame(self.confusion, index=['y = 0', 'y = 1'],
			columns=['y_pred = 0', 'y_pred = 1'])
		sn.set(font_scale=1.4)
		heatmap = sn.heatmap(dataframe, annot=True, annot_kws={'size': 16})
		fig = heatmap.get_figure()
		if show:
			plt.show()
		if path is not None:
			file_name = os.path.join(path, 'confusion.pdf')
			fig.savefig(file_name)

class SeqRNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers):
		super(SeqRNN, self).__init__()
		self.hidden_size = hidden_size
		self.i2h = nn.Linear(in_features=input_size + hidden_size, 
			out_features=hidden_size)
		self.i2o = nn.Linear(in_features=input_size + hidden_size, 
			out_features=output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		# input ~ (batch, alphabet_size); hidden ~ (batch, hidden_size)
		# ~ (batch, alphabet_size + hidden_size)
		combined = torch.cat((input, hidden), dim=1)
		# ~ (batch, hidden_size)
		hidden = self.i2h(combined)
		# ~ (batch, output_size)
		out = self.i2o(combined)
		# ~ (batch, output_size)
		out = self.softmax(out)

		return out, hidden

	def init_hidden(self, batch):
		return torch.zeros(batch, self.hidden_size, device=device)

class SeqGRU(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers):
		super(SeqGRU, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers

		self.embed = nn.Embedding(input_size, hidden_size)
		# if using embedding, should be:
		#self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, n_layers)
		
		# w/o embedding:
		self.gru = nn.GRU(input_size, hidden_size, n_layers)
		self.fc = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward_old(self, input, hidden):
		# input ~ (batch, alphabet)
		# hidden ~ (n_layers * directions, batch, hidden_size)
		input = input.unsqueeze(0) # make input ~ (1, batch, alphabet)

		# embedded ~ (batch, hidden_size)
		#embedded = self.embed(input)
		
		# out ~ (seqLen=1, batch, hidden)
		out, hidden = self.gru(input, hidden)
		
		# out ~ (seqLen=1, batch, output_size)
		out = self.fc(out)

		# out ~ (seqLen=1, batch, output_size) --> out ~ (batch, output_size)
		out = self.softmax(out.squeeze(0))

		return out, hidden

	def forward(self, input, hidden):
		# input ~ (max_seqlen, batch, alphabet)
		# hidden ~ (n_layers * directions, batch, hidden_size)

		# embedded ~ (batch, hidden_size)
		#embedded = self.embed(input)
		
		# out ~ (max_seqlen, batch, hidden)
		out, hidden = self.gru(input, hidden)
		#print("gru_out.size = ", out.size())
		
		# out ~ (max_seqlen, batch, output_size)
		out = self.fc(out)
		#print("fc_out.size() = ", out.size())

		# out ~ (seqLen, batch, output_size) --> out ~ (batch, output_size)
		out = self.softmax(out)
		#print("softmax_out = ", out)
		#print("softmax_out.size() = ", out.size())

		return out[-1], hidden

	def init_hidden(self, batch):
		return torch.zeros(self.n_layers, batch, self.hidden_size, device=device)

class SeqLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers):
		super(SeqLSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers

		self.embed = nn.Embedding(input_size, hidden_size)

		# w/o embedding:
		self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
		self.fc = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		input = input.unsqueeze(0)
		out, hidden = self.lstm(input, hidden)
		out = self.fc(out)
		out = self.softmax(out.squeeze(0))
		return out, hidden

	def init_hidden(self, batch):
		hidden_state = torch.zeros(self.n_layers, batch, self.hidden_size, device=device)
		cell_state = torch.zeros(self.n_layers, batch, self.hidden_size, device=device)
		return (hidden_state, cell_state)

class BetterLSTM(nn.Module):
	# input_size = alphabet_size
	def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers=1, bidir=False):
		super(BetterLSTM, self).__init__()
		self.input_size = input_size
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.num_dir = 2 if bidir else 1

		self.embed = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)

		# w/o embedding
		self.lstm = nn.LSTM(input_size=embedding_size, 
			hidden_size=hidden_size, num_layers=n_layers, bidirectional=bidir)
		self.fully_connected = nn.Linear(hidden_size, output_size)
		self.softmax = self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, h0, c0):
		embedded = self.embed(input)
		output, (h_final, c_final) = self.lstm(embedded, (h0, c0))
		out = self.fully_connected(h_final[-1])
		
		return out

	def init_hidden(self, batch):
		h0 = torch.zeros(self.n_layers * self.num_dir, 
			batch, self.hidden_size, device=device)
		c0 = torch.zeros(self.n_layers * self.num_dir, 
			batch, self.hidden_size, device=device)
		return h0, c0

def train_sample(model, optimizer, loss_function, seq_tensor, label_tensor):
	hidden = model.init_hidden(batch=1)
	optimizer.zero_grad()
	for i in range(seq_tensor.size()[0]):
		output, hidden = model(seq_tensor[i], hidden)
	loss = loss_function(output, label_tensor)
	loss.backward()
	nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
	optimizer.step()

	return output, loss.item()

def batch_train(padded_xs, y_batch, model, opt, loss_function):
	hidden = model.init_hidden(BATCH)
	opt.zero_grad()
	out, hidden = model(padded_xs, hidden)
	#print("out = ", out)
	#print("y_batch = ", y_batch)
	loss = loss_function(out, y_batch)
	loss.backward()
	#nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
	#print("loss = ", loss.item())
	return out, loss.item()	

def predict(model, sequence):
	with torch.no_grad():
		hidden = model.init_hidden(batch=1)
		for i in range(sequence.size()[0]):
			output, hidden = model(sequence[i], hidden)
		topind = output.argmax(dim=-1)
	return topind.item()

def get_accuracy(model, samples, labels):
	num_samples = len(samples)
	assert num_samples == len(labels)
	correct = 0
	for x, y in zip(samples, labels):
		y_pred = predict(model, x)
		if y_pred == y.item():
			correct += 1
	accuracy = correct / num_samples
	return accuracy * 100

def evaluate(model, samples, labels):
	num_samples = len(samples)
	assert num_samples == len(labels)
	correct = 0
	with torch.no_grad():
		for x, y in zip(samples, labels):
			x = x.unsqueeze(1)
			hidden = model.init_hidden(batch=1)
			out, _ = model(x, hidden)
			y_pred = out.argmax(dim=-1).item()
			if y_pred == y.item(): correct += 1
	return (correct / num_samples) * 100

def lstm_evaluate(model, samples, labels):
	num_samples = len(samples)
	assert num_samples == len(labels)
	correct = 0
	true_ys = []
	scores = []
	predictions = []
	true_neg = false_neg = false_pos = true_pos = 0
	num_neg = num_pos = 0

	with torch.no_grad():
		for x, y in zip(samples, labels):
			x = x.unsqueeze(1)
			h0, c0 = model.init_hidden(batch=1)
			out = model(x, h0, c0)
			scores.append(torch.max(out).item())
			y_pred = out.argmax(dim=-1).item()
			predictions.append(y_pred)
			y_true = y.item()
			true_ys.append(y_true)
			if y_pred == y_true: correct += 1

	accuracy = (correct / num_samples) * 100
	confusion = metrics.confusion_matrix(true_ys, predictions)

	'''metrics.roc_curve returns: 
	* fpr = array of increasing false pos rates
	* tpr = array of increasing true pos rates
	* thresholds = array of decreasing thresholds on the decision function
		used to compute the fpr and tpr
	'''
	increasing_fprs, increasing_tprs, thresholds = metrics.roc_curve(true_ys, scores)
	auc = metrics.roc_auc_score(true_ys, scores)
	
	return accuracy, increasing_fprs, increasing_tprs

def main():
	parameters = {}
	dataset = Dataset(dict_file, train_file, test_file, use_cuda)
	xtrain, ytrain = dataset.xtrain, dataset.ytrain
	xtest, ytest = dataset.xtest, dataset.ytest
	alphabet_size = dataset.alphabet_size
	hidden_size = args.hidden
	num_train, num_test = len(xtrain), len(xtest)
	print('training size: %d' % num_train)
	print('test size: %d' % num_test)
	print('device = %s' % device)
	num_classes = 2

	parameters['algorithm'] = 'lstm'
	parameters['train set'] = train_file
	parameters['test set'] = test_file
	parameters['dictionary file'] = dict_file
	parameters['cuda'] = use_cuda
	parameters['bidir'] = bidir
	parameters['layers'] = n_layers
	parameters['embedding size'] = embed_size
	parameters['learning rate'] = lr

	model = BetterLSTM(input_size=alphabet_size, embedding_size=embed_size,
		hidden_size=hidden_size, output_size=num_classes,
		n_layers=n_layers, bidir=bidir).to(device)

	if args.opt == 'adagrad':
		opt = optim.Adagrad(model.parameters(), lr=lr)
	elif args.opt == 'adam':
		opt = optim.Adam(model.parameters(), lr=lr)
	else:
		opt = optim.SGD(model.parameters(), lr=lr)

	loss_function = F.cross_entropy

	parameters['optimizer'] = args.opt
	parameters['loss function'] = 'cross entropy'

	iters = args.iters
	log_interval = args.log_interval

	parameters['iterations'] = iters

	hist = History()
	interval_loss = 0

	for i in trange(1, iters + 1):
		opt.zero_grad()
		rand = random.randint(0, num_train - 1)
		x, y = xtrain[rand], ytrain[rand]
		x = torch.unsqueeze(x, dim=1) # (seqlen, sigma) --> (seqlen, batch=1, sigma)
		h0, c0 = model.init_hidden(batch=1)
		y_pred = model(x, h0, c0)
		loss = loss_function(y_pred, y)
		loss.backward()
		opt.step()

		interval_loss += loss.item()

		if i % log_interval == 0:
			avg_loss = interval_loss / log_interval
			hist.add_loss(i, avg_loss)
			interval_loss = 0
			
			train_eval = Evaluation(model, dataset.xtrain, dataset.ytrain)
			test_eval = Evaluation(model, dataset.xtest, dataset.ytest)

			hist.add_acc(i, train_eval.accuracy, test_eval.accuracy)
			hist.add_auc(i, train_eval.auc, test_eval.auc)
			hist.add_roc_info(test_eval.increasing_fprs, 
				test_eval.increasing_tprs)
			summary = ("Iter {}\ntrain acc = {}\ntest acc = {}\n"
				"TPR/sensitvity/recall = {}\nTNR/specificity = {}\n"
				"train loss = {}\nAUC-ROC = {}".format(i, train_eval.accuracy, 
					test_eval.accuracy, test_eval.tpr, test_eval.tnr,
					avg_loss, test_eval.auc))
			print(summary)
			print("Confusion:")
			test_eval.show_confusion()

	final_train_eval = Evaluation(model, dataset.xtrain, dataset.ytrain)
	final_test_eval = Evaluation(model, dataset.xtest, dataset.ytest)

	summary = ("Final Eval:\ntrain acc = {}\ntest acc = {}\n"
		"TPR/sensitvity/recall = {}\nTNR/specificity = {}"
		"\nAUC-ROC = {}".format(train_eval.accuracy, 
		test_eval.accuracy, test_eval.tpr, test_eval.tnr, test_eval.auc))
	print(summary)

	final_test_eval.show_confusion()

	parameters['accuracy test'] = final_test_eval.accuracy
	parameters['accuracy train'] = final_train_eval.accuracy
	parameters['auc test'] = final_test_eval.auc
	parameters['auc train'] = final_train_eval.auc
	parameters['TPR/sensitvity/recall'] = final_test_eval.tpr
	parameters['TNR/specificity'] = final_test_eval.tnr

	# if output_directory specified, write data for future viewing
	# otherwise, it'll be discarded
	if args.output_directory is not None:
		path = args.output_directory
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path)
		file_prefix = "model"
		file_prefix = os.path.join(path, file_prefix)
		hist.save_data(file_prefix)
		summary_file = os.path.join(path, 'about.txt')
		
		with open(summary_file, 'w+') as f:
			now = datetime.datetime.now()
			f.write(now.strftime('%Y-%m-%d %H:%M') + '\n')
			f.write('command_used: python ' + ' '.join(sys.argv) + '\n')
			f.write('(may have included CUDA_VISIBLE_DEVICES=x first)\n\n')
			for key, value in sorted(parameters.items()):
				f.write(key + ': ' + str(value) + '\n')

		hist.plot_acc(show=False, path=path)
		hist.plot_loss(show=False, path=path)
		hist.plot_auc(show=False, path=path)
		hist.plot_roc(show=False, path=path)
		final_test_eval.plot_confusion(show=False, path=path)

	if args.show_graphs:
		hist.plot_acc(show=True)
		hist.plot_loss(show=True)
		hist.plot_auc(show=True)
		hist.plot_roc(show=True)
		final_test_eval.plot_confusion(show=True)

if __name__ == '__main__':
	main()

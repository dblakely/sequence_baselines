### dataset.py
### A class for containing datasets used by RNN baseline models
### Derrick Blakely, December 2018

import numpy as np
import torch
import random

class Dataset(object):
	
	def __init__(self, dictionary_file, train_file, test_file, use_cuda=False):
		self.device = torch.device('cuda' if use_cuda else 'cpu')
		self.dictionary = self.get_dict(dictionary_file)
		self.alphabet_size = len(self.dictionary)
		assert self.alphabet_size > 0
		self.xtrain, self.ytrain = self.prepare_data(train_file)
		self.xtest, self.ytest = self.prepare_data(test_file)
		self.n_train = len(self.xtrain)
		self.n_test = len(self.xtest)

	def get_dict(self, dict_file):
		dictionary = {}
		num = 0
		with open(dict_file, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip().lower()
				assert len(line) == 1
				assert line not in dictionary
				dictionary[line] = num
				num += 1
		return dictionary

	def char_to_index(self, char):
		return self.dictionary[char]

	# Create a one-hot encoding of a single characeter
	def char_to_tensor(self, char):
		tensor = torch.zeros(1, self.alphabet_size)
		tensor[0][self.char_to_index(char)] = 1
		return tensor

	def seq_to_tensor(self, seq):
		if type(seq) is str: seq = list(seq)
		'''
		tensor = torch.zeros(len(seq), 1, self.alphabet_size, device=self.device)
		for i, char in enumerate(seq):
			tensor[i][0][self.char_to_index(char)] = 1
		'''
		tensor = torch.zeros(len(seq), self.alphabet_size, device=self.device)
		for i, char in enumerate(seq):
			tensor[i][self.char_to_index(char)] = 1
		return tensor

	# obtain a list of one-hot tensors and a list of their labels
	def prepare_data(self, datafile):
		one_hot_seqs = []
		labels = []
		with open(datafile, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip().lower().replace(' ', '')
				length = len(line)
				if line[0] == '>':
					assert length == 2
					label = int(line[1])
					label_tensor = torch.tensor([label], dtype=torch.long, device=self.device)
					labels.append(label_tensor)
				else:
					one_hot_seq = self.seq_to_tensor(line)
					one_hot_seqs.append(one_hot_seq)
		assert len(one_hot_seqs) == len(labels)
		return one_hot_seqs, labels

	def get_batch(self, batch, training_data=True):
		xbatch = []
		ybatch = []
		max_idx = self.n_train - 1 if training_data else self.n_test - 1
		for _ in range(batch):
			rand = random.randint(0, max_idx)
			xbatch.append(self.xtrain[rand] if training_data else self.xtest[rand])
			ybatch.append(self.ytrain[rand] if training_data else self.ytest[rand])
		return xbatch, ybatch

import torch
from torch.utils import data

class Vocabulary(object):
	"""A class for storing the vocabulary of a 
	sequence dataset. Maps words or characters to indexes in the
	vocabulary.
	"""
	def __init__(self):
		self.token2idx = {}
		self.idx2token = {}
		self.size = len(self.token2idx)

	def add(self, token):
		"""
		Add a token to the vocabulary.
		Args:
			token: a letter (for char-level model) or word (for word-level model)
			for which to create a mapping to an integer (the idx).
		Return: 
			the index of the word. If it's already present, return its
			index. Otherwise, add it before returning the index.
		"""
		if token not in self.token2idx:
			self.token2idx[token] = self.size
			self.token2idx[self.size] = token
			self.size += 1
		return self.token2idx.get(token)


class FastaDataset(data.Dataset):
	"""Dataset class for creating FASTA-formatted
	sequence datasets.
	Args:
		file_path (string): path to the fasta file
		vocab (Vocabulary): a predefined vocabulary to use. Recommended if
			the dataset represents a test set that should have the exact
			same vocabulary as the training set.
		transform (callable, optional): A function/transform that takes in an
			:obj:`torch_geometric.data.Data` object and returns a transformed
			version. The data object will be transformed before every access.
			(default: :obj:`None`)
	"""

	def __init__(self, file_path, vocab=None, transform=None):
		self.file_path = file_path
		self.transform = transform
		self._vocab = Vocabulary() if vocab is None else vocab
		self.sequences = []
		self.labels = []
		self._process()

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		sequence = self.sequences[idx]
		sequence = sequence if self.transform is None else self.transform(sequence)
		label = self.labels[idx]
		return sequence, label

	def _process(self):
		"""Read a file in FASTA format. Specifically, resembles:
				>0
				ATCG
			where the first line is assumed to be a label line.
			Will read from self._file_path and store sequences in 
			self.sequences and labels in self.labels.
		"""
		with open(self.file_path, 'r', encoding='utf-8') as f:
			label_line = True
			for line in f:
				line = line.strip().lower()
				if label_line:
					split = line.split('>')
					assert len(split) == 2
					label = int(split[1])
					assert label in [-1, 0, 1]
					label = torch.tensor([label], dtype=torch.long)
					self.labels.append(label)
					label_line = False
				else:
					seq = list(line)
					seq = [self._vocab.add(token) for token in seq]
					seq = torch.tensor(seq, dtype=torch.long)
					self.sequences.append(seq)
					label_line = True
		assert len(self.sequences) == len(self.labels)

	def get_vocab(self):
		return self._vocab


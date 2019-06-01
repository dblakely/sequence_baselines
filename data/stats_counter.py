import os
import re

stats_file = "stats.txt"
excluded = ['webkb-test.fasta', 'webkb-train.fasta']
stats = []

for filename in os.listdir('./'):
	if filename.endswith(".fasta") and filename not in excluded:
		with open(filename, 'r') as f:
			labels = re.findall('>.\n', f.read(), re.MULTILINE)
			total = len(labels)
			labels = ''.join(labels)
			num_neg = len(re.findall('>0', labels))
			num_pos = len(re.findall('>1', labels))
			if (num_pos + num_neg != total):
				raise AssertionError("Error {}".format(filename))
			stats.append("{}\t\t\t{}\t{}\t{}\n".format(filename, num_pos, num_neg, total))
stats = sorted(stats)
with open(stats_file, 'w+') as f:
	f.write(''.join(stats))

from collections import Counter
from itertools import chain
import random
import os
import re

import numpy as np
from nltk.tokenize import word_tokenize
from contk.dataloader import Dataloader, LanguageGeneration
from contk._utils import trim_before_target
from contk.metric import MetricChain, PerlplexityMetric, \
						LanguageGenerationRecorder

from ..metric import LanguageGenerationProbabilityRecorder

class RenminDailyNews(LanguageGeneration):
	def __init__(self, file_path, min_vocab_times=3, max_sen_length=70):
		self._file_path = file_path
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		super().__init__(key_name=['train', 'valid', 'test'])

	def _loadset(self, path):
		files = os.listdir(path)
		all_sen = []
		for fn in files:
			with open(path + "/" + fn, encoding="gbk") as f:
				for line in f:
					utts = re.sub(R'\[(?P<c>.*?)\]\w+\b', lambda m: m.group('c'), line).split()
					utts = list(map(lambda x: x.split('/')[0], utts))

					lastpos = 0
					while lastpos < len(utts):
						nextpos = lastpos
						while nextpos < len(utts) and utts[nextpos] not in ["。", "？", "！"]:
							nextpos += 1
						all_sen.append(utts[lastpos: nextpos + 1])
						lastpos = nextpos + 1
		return all_sen

	def _load_data(self):
		r'''Loading dataset, invoked by LanguageGeneration.__init__
		'''
		origin_data = {}
		for key in self.key_name:
			origin_data[key] = self._loadset("%s/%s" % (self._file_path, key))

		vocab = list(chain(*(origin_data['train'])))
		# Important: Sort the words preventing the index changes between different runs
		vocab = sorted(Counter(vocab).most_common(), key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
		vocab_list = self.ext_vocab + list(map(lambda x: x[0], left_vocab))
		word2id = {w: i for i, w in enumerate(vocab_list)}
		print("vocab list length = %d" % len(vocab_list))

		line2id = lambda line: ([self.go_id] + \
					list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) + \
					[self.eos_id])[:self._max_sen_length]

		data = {}
		for key in self.key_name:
			data[key] = {}

			data[key] = list(map(line2id, origin_data[key]))
			vocab = list(chain(*(origin_data[key])))
			vocab_num = len(vocab)
			oov_num = len(list(filter(lambda word: word not in word2id, vocab)))
			length = list(map(len, origin_data[key]))
			cut_num = np.sum(np.maximum(np.array(length) - self._max_sen_length + 1, 0))
			print("%s set. OOV rate: %f, max length before cut: %d, cut word rate: %f" % \
					(key, oov_num / vocab_num, max(length), cut_num / vocab_num))
		return vocab_list, data

	def get_metric(self, gen_prob_key="gen_prob"):
		metric = MetricChain()
		metric.add_metric(PerlplexityMetric(self,
								 data_key='sentence',
								 data_len_key='sentence_length',
								 gen_prob_key=gen_prob_key))
		metric.add_metric(LanguageGenerationRecorder(self, sentence_key='sentence'))
		metric.add_metric(LanguageGenerationProbabilityRecorder(self, sentence_len_key='sentence_length', gen_prob_key=gen_prob_key))
		return metric

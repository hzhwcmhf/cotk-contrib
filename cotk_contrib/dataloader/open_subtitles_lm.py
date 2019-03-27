'''
A module for single turn dialog.
'''
from collections import Counter
from itertools import chain

import numpy as np

from cotk._utils.unordered_hash import UnorderedSha256
from cotk._utils.file_utils import get_resource_file_path
from cotk.dataloader import LanguageGeneration
from cotk.metric import MetricChain, PerplexityMetric, BleuCorpusMetric, SingleTurnDialogRecorder, \
			HashValueRecorder

class OpenSubtitlesLM(LanguageGeneration):
	'''A dataloader for OpenSubtitles dataset.

	Arguments:
		file_id (str): a str indicates the source of OpenSubtitles dataset.
		file_type (str): a str indicates the type of OpenSubtitles dataset. Default: "OpenSubtitles"
		min_vocab_times (int): A cut-off threshold of `UNK` tokens. All tokens appear
			less than `min_vocab_times`	will be replaced by `<unk>`. Default: 10.
		max_sen_length (int): All sentences longer than `max_sen_length` will be shortened
			to first `max_sen_length` tokens. Default: 50.
		invalid_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
			not less than `invalid_vocab_times` in the **whole dataset** (except valid words) will be
			marked as invalid vocabs. Otherwise, they are unknown words, both in training or
			testing stages. Default: 0 (No unknown words).

	Refer to :class:`.SingleTurnDialog` for attributes and methods.

	References:
		[1] http://opus.nlpl.eu/OpenSubtitles.php

		[2] P. Lison and J. Tiedemann, OpenSubtitles2016: Extracting Large Parallel Corpora from
		Movie and TV Subtitles.(LREC 2016)
	'''
	def __init__(self, file_id, file_type="OpenSubtitles", min_vocab_times=10, \
			max_sen_length=50, invalid_vocab_times=0):
		self._file_id = file_id
		self._file_path = get_resource_file_path(file_id, file_type)
		self._file_type = file_type
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		self._invalid_vocab_times = invalid_vocab_times
		super().__init__()

	def _load_data(self):
		r'''Loading dataset, invoked by `SingleTurnDialog.__init__`
		'''
		origin_data = {}
		for key in self.key_name:
			f_file = open("%s/opensub_pair_%s.post" % (self._file_path, key))
			g_file = open("%s/opensub_pair_%s.response" % (self._file_path, key))
			origin_data[key] = {}
			post = list(map(lambda line: line.split(), f_file.readlines()))
			resp = list(map(lambda line: line.split(), g_file.readlines()))
			origin_data[key]['sent'] = list(map(lambda x: x[0] + x[1], zip(post, resp)))

		raw_vocab_list = list(chain(*(origin_data['train']['sent'])))
		# Important: Sort the words preventing the index changes between different runs
		vocab = sorted(Counter(raw_vocab_list).most_common(), key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
		vocab_list = self.ext_vocab + list(map(lambda x: x[0], left_vocab))
		valid_vocab_len = len(vocab_list)
		valid_vocab_set = set(vocab_list)

		for key in self.key_name:
			if key == 'train':
				continue
			raw_vocab_list.extend(list(chain(*(origin_data[key]['sent']))))
		vocab = sorted(Counter(raw_vocab_list).most_common(), \
					   key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list( \
			filter( \
				lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, \
				vocab))
		vocab_list.extend(list(map(lambda x: x[0], left_vocab)))

		print("valid vocab list length = %d" % valid_vocab_len)
		print("vocab list length = %d" % len(vocab_list))

		word2id = {w: i for i, w in enumerate(vocab_list)}
		line2id = lambda line: ([self.go_id] + \
					list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) + \
					[self.eos_id])[:self._max_sen_length]

		data = {}
		data_size = {}
		for key in self.key_name:
			data[key] = {}

			data[key]['sent'] = list(map(line2id, origin_data[key]['sent']))
			data_size[key] = len(data[key]['sent'])
			vocab = list(chain(*(origin_data[key]['sent'])))
			vocab_num = len(vocab)
			oov_num = len(list(filter(lambda word: word not in word2id, vocab)))
			invalid_num = len( \
				list( \
					filter( \
						lambda word: word not in valid_vocab_set, \
						vocab))) - oov_num
			length = list(map(len, origin_data[key]['sent']))
			cut_num = np.sum(np.maximum(np.array(length) - self._max_sen_length + 1, 0))
			print("%s set. invalid rate: %f, unknown rate: %f, max length before cut: %d, \
					cut word rate: %f" % \
					(key, invalid_num / vocab_num, oov_num / vocab_num, max(length), cut_num / vocab_num))
		return vocab_list, valid_vocab_len, data, data_size

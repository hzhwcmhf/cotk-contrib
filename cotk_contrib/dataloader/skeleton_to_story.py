from itertools import chain
from collections import Counter
import random

import numpy as np
from cotk.dataloader import SingleTurnDialog
from cotk._utils.file_utils import get_resource_file_path

class SkeletonToStory(SingleTurnDialog):
	def __init__(self, file_id, min_vocab_times=10,
			max_sen_length=50, invalid_vocab_times=0,
			validate_division_seed=0, validate_k_fold=10):
		self._file_id = file_id
		self._file_path = get_resource_file_path(file_id)
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		self._invalid_vocab_times = invalid_vocab_times
		self._validate_division_seed = validate_division_seed
		self._validate_k_fold = validate_k_fold
		super().__init__()

	def _read_file(self, path):
		f_file = open(path)
		data = []
		now_story = []
		for line in f_file:
			if line.strip() == "------":
				data.append(now_story)
				now_story = []
			else:
				now_story.extend(line.strip().split())
				now_story.extend(".")
		return data

	def _load_data(self):
		r'''Loading dataset, invoked by `LanguageGeneration.__init__`
		'''

		skeleton_data = self._read_file("%s/skeleton.txt" % (self._file_path))
		story_data = self._read_file("%s/story.txt" % (self._file_path))
		all_data = list(zip(skeleton_data, story_data))
		random_state = random.getstate()
		random.seed(self._validate_division_seed)
		random.shuffle(all_data)
		dev_data_len = len(all_data) // (self._validate_k_fold + 1)
		train_data, dev_data, test_data = \
				all_data[dev_data_len * 2:], all_data[:dev_data_len], all_data[dev_data_len:dev_data_len*2]
		random.setstate(random_state)

		print("train sentence num: %d, dev: %d, test: %d" % (len(train_data), len(dev_data), len(test_data)))

		def turn_pair_to_dict(data):
			pair_data = list(zip(*data))
			return {"post": pair_data[0], "resp": pair_data[1]}

		self.origin_data = origin_data = {
			"train": turn_pair_to_dict(train_data),
			"dev": turn_pair_to_dict(dev_data),
			"test": turn_pair_to_dict(test_data)
		}

		raw_vocab_list = list(chain(*(origin_data['train']['post']))) + list(chain(*(origin_data['train']['resp'])))
		# Important: Sort the words preventing the index changes between
		# different runs
		vocab = sorted(Counter(raw_vocab_list).most_common(), \
						key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list( \
			filter( \
				lambda x: x[1] >= self._min_vocab_times, \
				vocab))
		vocab_list = self.ext_vocab + list(map(lambda x: x[0], left_vocab))
		valid_vocab_len = len(vocab_list)
		valid_vocab_set = set(vocab_list)

		for key in self.key_name:
			if key == 'train':
				continue
			raw_vocab_list.extend(list(chain(*(origin_data[key]['post']))))
			raw_vocab_list.extend(list(chain(*(origin_data[key]['resp']))))
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
		def line2id(line):
			return ([self.go_id] + \
					list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) \
					+ [self.eos_id])[:self._max_sen_length]

		data = {}
		data_size = {}
		for key in self.key_name:
			data[key] = {}

			data[key]['post'] = list(map(line2id, origin_data[key]['post']))
			data[key]['resp'] = list(map(line2id, origin_data[key]['resp']))
			data_size[key] = len(data[key]['post'])
			vocab = list(chain(*(origin_data[key]['post'] + origin_data[key]['resp'])))
			vocab_num = len(vocab)
			oov_num = len(list(filter(lambda word: word not in word2id, vocab)))
			invalid_num = len( \
				list( \
					filter( \
						lambda word: word not in valid_vocab_set, \
						vocab))) - oov_num
			length = list(map(len, origin_data[key]['post'] + origin_data[key]['resp']))
			cut_num = np.sum(np.maximum(np.array(length) - self._max_sen_length + 1, 0))
			print("%s set. invalid rate: %f, unknown rate: %f, max length before cut: %d, \
					cut word rate: %f" % \
					(key, invalid_num / vocab_num, oov_num / vocab_num, max(length), cut_num / vocab_num))
		return vocab_list, valid_vocab_len, data, data_size


#if __name__ == "__main__":
#	dm = SkeletonGeneration("./skeleton")

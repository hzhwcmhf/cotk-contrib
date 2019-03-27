from collections import Counter
from itertools import chain
import random

import numpy as np

from pytorch_pretrained_bert import BertTokenizer
from cotk._utils.unordered_hash import UnorderedSha256
from cotk._utils.file_utils import get_resource_file_path
from cotk.dataloader import Dataloader
from cotk.metric import MetricChain, MetricChain, PerplexityMetric, LanguageGenerationRecorder, \
	FwBwBleuCorpusMetric, SelfBleuCorpusMetric, HashValueRecorder

class LanguageGeneration_BERT(Dataloader):
	def __init__(self, key_name=None, bert_tokenize_name='bert-base-uncased'):
		super().__init__()

		self.key_name = key_name or ["train", "dev", "test"]

		self.tokenizer = tokenizer = BertTokenizer.from_pretrained(bert_tokenize_name)
		self.pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
		self.unk_id = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
		self.sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

		self.data = self._load_data()

		self.index = {}
		self.batch_id = {}
		self.batch_size = {}
		for key in self.key_name:
			self.batch_id[key] = 0
			self.batch_size[key] = None
			self.index[key] = list(range(len(self.data[key]['sent'])))

	def _load_data(self):
		raise NotImplementedError("This function should be implemented by subclasses.")

	@property
	def vocab_size(self):
		return len(self.tokenizer.vocab)

	def restart(self, key, batch_size=None, shuffle=True):
		'''Initialize mini-batches. Must call this function before :func:`get_next_batch`
		or an epoch is end.

		Arguments:
			key (str): must be contained in `key_name`
			batch_size (None or int): default (None): use last batch_size.
			shuffle (bool): whether to shuffle the data. default: `True`
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		if batch_size is None and self.batch_size[key] is None:
			raise ValueError("You need batch_size to initialize.")
		if shuffle:
			random.shuffle(self.index[key])

		self.batch_id[key] = 0
		if batch_size is not None:
			self.batch_size[key] = batch_size
		print("%s set restart, %d batches and %d left" % (key, \
				len(self.index[key]) // self.batch_size[key], \
				len(self.index[key]) % self.batch_size[key]))

	def get_batch(self, key, index):
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		res = {}
		batch_size = len(index)
		res["sent_length"] = np.array(list(map(lambda i: len(self.data[key]['sent'][i]), index)))
		res["sent"] = np.zeros((batch_size, np.max(res["sent_length"])), dtype=int)
		res["sent_mask"] = np.zeros((batch_size, np.max(res["sent_length"])), dtype=int)
		for i, j in enumerate(index):
			sent = self.data[key]['sent'][j]
			res["sent"][i, :len(sent)] = sent
			res["sent_mask"][i, :len(sent)] = 1
		return res

	def get_next_batch(self, key, ignore_left_samples=False):
		'''Get next batch.

		Arguments:
			key (str): must be contained in `key_name`
			ignore_left_samples (bool): Ignore the last batch, whose sample num
				is not equal to `batch_size`. Default: `False`

		Returns:
			A dict like :func:`get_batch`, or None if the epoch is end.
		'''
		if key not in self.key_name:
			raise ValueError("No set named %s." % key)
		if self.batch_size[key] is None:
			raise RuntimeError("Please run restart before calling this function.")
		batch_id = self.batch_id[key]
		start, end = batch_id * self.batch_size[key], (batch_id + 1) * self.batch_size[key]
		if start >= len(self.index[key]):
			return None
		if ignore_left_samples and end > len(self.index[key]):
			return None
		index = self.index[key][start:end]
		res = self.get_batch(key, index)
		self.batch_id[key] += 1
		return res

	def sen_to_index(self, sen):
		return self.tokenizer.convert_tokens_to_ids(sen)

	def trim_index(self, index):
		idx = len(index)
		while index[idx-1] == self.pad_id:
			idx -= 1
		index = index[:idx]
		return index

	def index_to_sen(self, index, trim=True):
		if trim:
			index = self.trim_index(index)
		return self.tokenizer.convert_ids_to_tokens(index)

	def get_teacher_forcing_metric(self, gen_log_prob_key="gen_log_prob"):
		'''Get metric for teacher-forcing mode.

		It contains:

		* :class:`.metric.PerplexityMetric`

		Arguments:
				gen_prob_key (str): default: `gen_prob`. Refer to :class:`.metric.PerplexityMetric`
		'''
		metric = MetricChain()
		metric.add_metric(HashValueRecorder(hash_key="teacher_forcing_hashvalue"))
		metric.add_metric(PerplexityMetric(self, \
					reference_allvocabs_key='sent_allvocabs', \
					reference_len_key='sent_length', \
					gen_log_prob_key=gen_log_prob_key))
		return metric

	def get_inference_metric(self, gen_key="gen"):
		metric = MetricChain()
		metric.add_metric(LanguageGenerationRecorder(self, gen_key=gen_key))
		return metric

	def add_sep(self, sent):
		return ("[CLS] " + sent + " [SEP]").replace("  ", " ")

class MSCOCO_BERT(LanguageGeneration_BERT):
	def __init__(self, file_id, file_type="MSCOCO", max_sen_length=70, bert_tokenize_name='bert-base-uncased'):
		self._file_id = file_id
		self._file_path = get_resource_file_path(file_id, file_type)
		self._max_sen_length = max_sen_length
		super().__init__(bert_tokenize_name=bert_tokenize_name)

	def _load_data(self):
		origin_data = {}
		for key in self.key_name:
			f_file = open("%s/mscoco_%s.txt" % (self._file_path, key))
			origin_data[key] = {}
			origin_data[key]['sent'] = list( \
				map(lambda line: self.tokenizer.tokenize(self.add_sep(line.lower())), f_file.readlines()))

		data = {}
		data_size = {}
		for key in self.key_name:
			data[key] = {}
			data[key]['sent'] = list(map(self.tokenizer.convert_tokens_to_ids, origin_data[key]['sent']))
			data_size[key] = len(data[key]['sent'])

			vocab = list(chain(*(origin_data[key]['sent'])))
			vocab_num = len(vocab)
			length = list( \
				map(len, origin_data[key]['sent']))
			cut_num = np.sum( \
				np.maximum( \
					np.array(length) - \
					self._max_sen_length + \
					1, \
					0))
			print( \
				"%s set. max length before cut: %d, cut word rate: %f" % \
				(key, max(length), cut_num / vocab_num))
		return data, data_size

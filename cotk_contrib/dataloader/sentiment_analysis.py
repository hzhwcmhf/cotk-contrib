from collections import Counter
from itertools import chain
import random

import xml.etree.ElementTree as ET
import numpy as np
from nltk.tokenize import word_tokenize
from cotk.dataloader import Dataloader
from cotk._utils import trim_before_target
from cotk.metric import MetricChain

from ..metric import AspectBasedSentimentAnalysisMetric, \
					AspectBasedSentimentAnalysisHardMetric, \
					AspectBasedSentimentAnalysisOutofDomainMetric

class SentimentAnalysis(Dataloader):
	def __init__(self,		\
			ext_vocab=None, \
			key_name=None,	\
		):
		super().__init__()

		# initialize by default value. (can be overwritten by subclass)
		self.ext_vocab = ext_vocab or ["<pad>", "<unk>", "<go>", "<eos>"]
		self.pad_id = self.ext_vocab.index("<pad>")
		self.unk_id = self.ext_vocab.index("<unk>")
		self.go_id = self.ext_vocab.index("<go>")
		self.eos_id = self.ext_vocab.index("<eos>")
		self.key_name = key_name or ["train", "dev", "test"]

		# initialize by subclass
		self.vocab_list, self.data = self._load_data()
		self.word2id = {w: i for i, w in enumerate(self.vocab_list)}

		# postprocess initialization
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
		'''Equals to `len(self.vocab_list)`. Read only.
		'''
		return len(self.vocab_list)

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
			raise ValueError("You need batch_size to intialize.")
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
		res["low_aspect"] = np.array(list(map(lambda i: self.data[key]['low_aspect'][i], index)))
		res["high_aspect"] = np.array(list(map(lambda i: self.data[key]['high_aspect'][i], index)))
		res['hint'] = list(map(lambda i: self.data[key]['hint'][i], index))
		for i, j in enumerate(index):
			sent = self.data[key]['sent'][j]
			res["sent"][i, :len(sent)] = sent
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
		'''Convert a sentences from string to index representation.

		Arguments:
			sen (list): a list of str, representing each token of the sentences.

		Examples:
			>>> dataloader.sen_to_index(
			...		["<go>", "I", "have", "been", "to", "Sichuan", "province", "eos"])
			>>>

		Todo:
			* fix the missing example
		'''
		return list(map(lambda word: self.word2id.get(word, self.unk_id), sen))

	def trim_index(self, index):
		'''Trim index. There will be two steps:
			* find first `<eos>` and abondon words after it (included the `<eos>`).
			* ignore `<pad>` s at the end of the sentence.

		Arguments:
			index (list): a list of int

		Examples:
			>>> dataloader.index_to_sen(
			...		[])
			>>>

		Todo:
			* fix the missing example
		'''

		#index = trim_before_target(list(index), self.eos_id)
		idx = len(index)
		while index[idx-1] == self.pad_id:
			idx -= 1
		index = index[:idx]
		return index

	def index_to_sen(self, index, trim=True):
		'''Convert a sentences from index to string representation

		Arguments:
			index (list): a list of int
			trim (bool): if True, call :func:`trim_index` before convertion.

		Examples:
			>>> dataloader.index_to_sen(
			...		[])
			>>>

		Todo:
			* fix the missing example
		'''
		if trim:
			index = self.trim_index(index)
		return list(map(lambda word: self.vocab_list[word], index))

	def get_metric(self):
		metric = MetricChain()
	# 	metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key))
	# 	metric.add_metric(SingleDialogRecorder(self, gen_key=gen_key))
		return metric

class SemEvalABSA(SentimentAnalysis):
	def __init__(self, file_path, train14=True, test14=True, min_vocab_times=3, max_sen_length=70, \
					   validate_division_seed=0, validate_k_fold=5):
		self._file_path = file_path
		self._train14 = train14
		self._test14 = test14
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		self._validate_division_seed = validate_division_seed
		self._validate_k_fold = validate_k_fold
		super().__init__()

	def _load_semeval2014(self, file_path):
		_mapping_from14to15 = {"anecdotes/miscellaneous#": "restaurant#miscellaneous",\
						"price#": "restaurant#prices"}

		data = ET.ElementTree(file=file_path).getroot()
		res = []
		for sentences in data:
			sample = {}
			for node in sentences:
				if node.tag == "text":
					sample["text"] = word_tokenize(node.text.lower())
				elif node.tag == "aspectCategories":
					sample["label"] = {}
					for aspect_node in node:
						category = aspect_node.attrib['category'].lower() + "#"
						category = _mapping_from14to15.get(category, category)
						sample["label"][category] = (aspect_node.attrib['polarity'], None, None, None)
			if "label" not in sample:
				raise ValueError("data error, category not found")
			res.append(sample)
		return res

	def _load_semeval2015_2016(self, file_path):
		data = ET.ElementTree(file=file_path).getroot()
		res = []
		for review_node in data:
			for sentence_node in review_node[0]:
				sample = {"label": {}}  # some text may don't have any label
				for node in sentence_node:
					if node.tag == "text":
						sample['origin_text'] = node.text.lower()
						sample["text"] = word_tokenize(node.text.lower().replace("&apos;", "'"))
					elif node.tag == "Opinions":
						for aspect_node in node:
							category = aspect_node.attrib['category'].lower()
							from_pos = int(aspect_node.attrib["from"])
							to_pos = int(aspect_node.attrib["to"])
							hint = sample['origin_text'][from_pos:to_pos]
							sample["label"][category] = \
								(aspect_node.attrib['polarity'], from_pos, to_pos, hint)
				res.append(sample)
		return res

	def _load_data(self):
		train_data = []
		test_data = []
		if self._train14:
			train_data.extend(self._load_semeval2014(self._file_path + \
												"/ABSA_SemEval2014/Restaurants_Train_Final.xml"))
		if self._test14:
			test_data.extend(self._load_semeval2014(self._file_path + \
												"/ABSA_SemEval2014/Restaurants_Test.xml"))
		train_data.extend(self._load_semeval2015_2016(self._file_path + \
											"/ABSA_SemEval2015/Restaurants_Train_Final.xml"))
		test_data.extend(self._load_semeval2015_2016(self._file_path + \
											"/ABSA_SemEval2015/Restaurants_Test.xml"))
		train_data.extend(self._load_semeval2015_2016(self._file_path + \
											"/ABSA_SemEval2016/Restaurants_Train_Final.xml"))
		test_data.extend(self._load_semeval2015_2016(self._file_path + \
											"/ABSA_SemEval2016/Restaurants_Test.xml"))

		random_state = random.getstate()
		random.seed(self._validate_division_seed)
		random.shuffle(train_data)
		dev_data_len = len(train_data) // self._validate_k_fold
		train_data, dev_data = train_data[dev_data_len:], train_data[:dev_data_len]
		random.setstate(random_state)

		self.origin_data = {"train": train_data, "dev": dev_data, "test": test_data}

		vocab = list(chain(*[sen['text'] for sen in train_data]))
		# Important: Sort the words preventing the index changes between different runs
		vocab = sorted(Counter(vocab).most_common(), key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
		vocab_list = self.ext_vocab + list(map(lambda x: x[0], left_vocab))
		word2id = {w: i for i, w in enumerate(vocab_list)}
		print("vocab list length = %d" % len(vocab_list))

		all_aspect = list(set(chain(*[list(sen["label"]) for sen in train_data])))
		low_aspect = filter(lambda x: x.split("#")[1], all_aspect)
		self.low_aspect = low_aspect = sorted(list(low_aspect))
		high_aspect = set(aspect.split("#")[0] for aspect in all_aspect)
		self.high_aspect = high_aspect = sorted(list(high_aspect))
		polarity = set(chain(*[list(map(lambda x: x[0], sen["label"].values())) for sen in train_data]))
		self.polarity = polarity = ["none"] + sorted(list(polarity))


		def _process(sample):
			now_data = {}
			now_data['sent'] = ([self.go_id] + \
						list(map(lambda word: word2id.get(word, self.unk_id), sample['text'])) + \
						[self.eos_id])[:self._max_sen_length]
			now_data['vocab_num'] = len(sample['text'])
			now_data['oov_num'] = len(list(filter(lambda word: word not in word2id, sample['text'])))
			now_data['cut_num'] = max(0, len(sample['text']) - self._max_sen_length + 1)
			now_data['low_aspect'] = np.zeros((self.low_aspect_size), dtype=int)
			now_data['high_aspect'] = np.zeros((self.high_aspect_size), dtype=int)
			now_data['hint'] = [None for i in range(self.low_aspect_size)]
			for low_label, (po, frompos, topos, hint) in sample['label'].items():
				high_label = low_label.split("#")[0]
				po_id = polarity.index(po)

				if low_label in low_aspect:
					low_id = low_aspect.index(low_label)
					now_data['low_aspect'][low_id] = po_id

				high_id = high_aspect.index(high_label)
				now_data['high_aspect'][high_id] = po_id
				#TODO: how to use frompos & topos
				now_data['hint'][low_id] = (frompos, topos, hint)
			return now_data

		data = {}
		for key in self.key_name:
			attr_list = ["sent", "low_aspect", "high_aspect", "vocab_num", "oov_num", "cut_num", "hint"]
			data[key] = {key: [] for key in attr_list}
			for sample in self.origin_data[key]:
				sample = _process(sample)
				for attr in attr_list:
					data[key][attr].append(sample[attr])
			oov_num = data[key]["oov_num"]
			vocab_num = data[key]["vocab_num"]
			cut_num = data[key]["cut_num"]
			print("%s set. OOV rate: %f, max length before cut: %d, cut word rate: %f" % \
					(key, sum(oov_num) / sum(vocab_num), max(vocab_num), sum(cut_num) / sum(vocab_num)))
		return vocab_list, data

	@property
	def low_aspect_size(self):
		return len(self.low_aspect)
	@property
	def high_aspect_size(self):
		return len(self.high_aspect)
	@property
	def polarity_size(self):
		return len(self.polarity)

	def get_metric(self):
		metric = MetricChain()
		metric.add_metric(AspectBasedSentimentAnalysisMetric(self))
		metric.add_metric(AspectBasedSentimentAnalysisHardMetric(self))
		metric.add_metric(AspectBasedSentimentAnalysisOutofDomainMetric(self))
		return metric

	def get_statistics(self):
		print("vocab size: %d" % self.vocab_size)
		print("aspect size: %d" % self.low_aspect_size)
		print("aspect: %s" % self.low_aspect)
		print("polarity size: %d" % self.polarity_size)
		print("polarity: %s" % self.polarity)

		def _get_statistics(key):
			print("Set %s" % key)
			# res["low_aspect"] = np.array(list(map(lambda i: self.data[key]['low_aspect'][i], index)))
			num = 0
			hard_num = 0
			for i in range(len(self.data[key]['low_aspect'])):
				label = self.data[key]['low_aspect'][i]
				num += 1
				hard_num += np.sum(np.bincount(label)[1:] != 0) > 1
			print("\tnum: %d" % num)
			print("\thard num: %d" % hard_num)

		_get_statistics("train")
		_get_statistics("dev")
		_get_statistics("test")
	
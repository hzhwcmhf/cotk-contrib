import json
from itertools import chain
from collections import Counter

import numpy as np
from cotk.dataloader import MultiTurnDialog, UbuntuCorpus
from cotk.metric import BleuPrecisionRecallMetric, EmbSimilarityPrecisionRecallMetric, MetricChain

class SwitchboardCorpus(MultiTurnDialog):
	'''A dataloader for Switchboard dataset.

	Arguments:
		file_id (str): a str indicates the source of SwitchboardCorpus dataset.
		{ARGUMENTS}

	Refer to :class:`.MultiTurnDialog` for attributes and methods.

	Todo:
		* add references
	'''

	ARGUMENTS = UbuntuCorpus.ARGUMENTS

	def __init__(self, file_path, min_vocab_times=5, max_sen_length=50, max_turn_length=1000, \
				 invalid_vocab_times=0):
		self._file_path = file_path
		self._min_vocab_times = min_vocab_times
		self._max_sen_length = max_sen_length
		self._max_turn_length = max_turn_length
		self._invalid_vocab_times = invalid_vocab_times

		self.word2id = {}
		super().__init__()

	def _convert2ids(self, origin_data):
		'''Convert topic, da, word to ids, invoked by ^SwitchboardCorpus._load_data^
		and ^SwitchboardCorpus._load_multi_ref_data^

		Arguments:
			origin_data (dict): Contains at least:

				* session (list): A 3-d list, utterances in words.
				  Size of the outermost list: num_data.
				  Size of the second innermost list: num of utterances in a session.
				  Size of the innermost list: num of words in a utterance.

		Returns:
			(dict): Contains:

			* session (list): utterances in ids. Size: same as input
		'''
		data = {}
		sess2id = lambda sess: [([self.go_id] + \
								 list(map(lambda word: self.word2id.get(word, self.unk_id), utt)) + \
								 [self.eos_id])[:self._max_sen_length] for utt in sess]
		data['session'] = list(map(sess2id, origin_data['session']))
		return data

	def _read_file(self, filepath, add_pre_turn=True, add_suf_turn=False):
		'''Reading data from file, invoked by ^SwitchboardCorpus._load_data^
			and ^SwitchboardCorpus._load_multi_ref_data^

		Arguments:
			* filepath (str): Name of the file to read from
			* add_pre_turn (bool): Whether to add turn ^<d>^ ahead of each session
		'''
		origin_data = {'session': []}
		with open(filepath, "r") as data_file:
			for line in data_file:
				line = json.loads(line)
				prefix_utts = [['X', '<d>']] + line['utts']
				# pylint: disable=cell-var-from-loop
				suffix_utts = list(map(lambda utt: utt[1][1].strip() + ' ' \
							if prefix_utts[utt[0]][0] == utt[1][0] \
							else '<eos> ' + utt[1][1].strip() + ' ', enumerate(line['utts'])))
				utts = ('<d> ' + "".join(suffix_utts).strip()).split("<eos>")
				sess = list(map(lambda utt: utt.strip().split(), utts))
				if not add_pre_turn:
					sess = sess[1:]
				if add_suf_turn:
					sess += [['<d>']]
				origin_data['session'].append(sess[:self._max_turn_length])
		return origin_data

	def _build_vocab(self, origin_data):
		# TODO: build vocab has to use multi_ref data
		r'''Building vocabulary(words, topics, da), invoked by `SwitchboardCorpus._load_data`
		'''
		raw_vocab = list(chain(*chain(*origin_data['train']['session'])))
		vocab = sorted(Counter(raw_vocab).most_common(), key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
		left_vocab = list(map(lambda x: x[0], left_vocab))
		vocab_list = self.ext_vocab + left_vocab
		self.valid_vocab_len = len(vocab_list)
		valid_vocab_set = set(vocab_list)

		for key in self.key_name:
			if key == 'train':
				continue
			raw_vocab.extend(list(chain(*chain(*(origin_data[key]['session'])))))
		vocab = sorted(Counter(raw_vocab).most_common(), key=lambda pair: (-pair[1], pair[0]))
		left_vocab = list(filter(lambda x: \
				x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, vocab))
		left_vocab = list(map(lambda x: x[0], left_vocab))
		vocab_list.extend(left_vocab)

		self.word2id = {w: i for i, w in enumerate(vocab_list)}

		print("valid vocab list length = %d" % self.valid_vocab_len)
		print("vocab list length = %d" % len(vocab_list))
		return vocab_list, valid_vocab_set

	def _load_multi_ref_data(self):
		r'''Loading dataset, invoked by `SwitchboardCorpus._load_data`
		'''
		filename = '%s/switchboard_corpus_multi_ref.jsonl' % self._file_path
		candidate = []
		with open(filename, "r") as data_file:
			idx = 0
			for line in data_file:
				line = json.loads(line)
				utt2id = lambda utt: list(map(lambda w: \
										self.word2id.get(w, self.unk_id), utt[1].strip().split()))
				candidate.append(list(map(utt2id, line['responses'])))
				idx += 1
		return candidate

	def _load_data(self):
		r'''Loading dataset, invoked by `MultiTurnDialog.__init__`
		'''
		origin_data = {}
		self.key_name.append('multi_ref')
		for key in self.key_name:
			origin_data[key] = self._read_file('%s/switchboard_corpus_%s.jsonl' % (self._file_path, key), \
											   add_pre_turn=(key != 'multi_ref'), \
											   add_suf_turn=(key == 'multi_ref'))

		vocab_list, valid_vocab_set = self._build_vocab(origin_data)

		data = {}
		data_size = {s: 0 for s in self.key_name}
		for key in self.key_name:
			data[key] = self._convert2ids(origin_data[key])
			data_size[key] = len(data[key]['session'])

			vocab = list(chain(*chain(*(origin_data[key]['session']))))
			vocab_num = len(vocab)
			oov_num = len(list(filter(lambda word: word not in self.word2id, vocab)))
			invalid_vocab_num = len(list(filter(lambda word: \
											word not in valid_vocab_set, vocab))) - oov_num
			sent_lens = list(map(len, chain(*origin_data[key]['session'])))
			cut_word_num = np.sum(np.maximum(np.array(sent_lens) - self._max_sen_length + 2, 0))
			turn_lens = list(map(len, origin_data[key]['session']))
			cut_sent_num = np.sum(np.maximum(np.array(turn_lens) - self._max_turn_length, 0))
			print(("%s set. invalid rate: %f, unknown rate: %f, max sentence length before cut: %d, " + \
				   "cut word rate: %f\n\tmax turn length before cut: %d, cut sentence rate: %f") % \
				  (key, invalid_vocab_num / vocab_num, oov_num / vocab_num, max(sent_lens), \
				   cut_word_num / vocab_num, max(turn_lens), cut_sent_num / np.sum(turn_lens)))
		data['multi_ref']['candidate'] = self._load_multi_ref_data()
		return vocab_list, len(valid_vocab_set), data, data_size

	def get_batch(self, key, index, needhash=False):
		'''Get a batch of specified `index`.

		Arguments:
			key (str): must be contained in `key_name`
			index (list): a list of specified index
			needhash (bool): whether to return a hashvalue
			  representing this batch of data. Default: False.

		Returns:
			(dict): A dict contains what is in the return of MultiTurnDialog.get_batch.
			  It additionally contains:

				* candidates (list): A 3-d list, multiple responses for reference
				  Size of outermost list: batch_size
				  Size of second innermost list: varying num of references
				  Size of innermost list: varying num of words in a reference

			See the example belows.

		Examples:
			>>> dataloader.get_batch('train', [1])
			>>>

		Todo:
			* fix the missing example
		'''
		res = super().get_batch(key, index)
		gather = lambda sub_key: [self.data[key][sub_key][i] for i in index]
		for sub_key in self.data[key]:
			if sub_key not in res:
				res[sub_key] = gather(sub_key) # TODO: candidates renamed to candidates_allvocabs
		#TODO: add hashvalue for SwitchBoard
		return res

	#TODO: renamed to inference metric. embedding should have a default realization (use wordvec from Glove)
	def get_precision_recall_metric(self, embed):
		'''Get metrics for precision and recall in terms of BLEU, cosine similarity.

		It contains:

		* :class:`.metric.PrecisionRecallMetric`

		Arguments:
			embed (:class:`numpy.array`): Glove word embedding
		'''
		metric = MetricChain()
		for ngram in range(1, 5):
			metric.add_metric(BleuPrecisionRecallMetric(self, ngram))
		metric.add_metric(EmbSimilarityPrecisionRecallMetric(self, embed, 'avg'))
		metric.add_metric(EmbSimilarityPrecisionRecallMetric(self, embed, 'extrema'))
		return metric

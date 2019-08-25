import tempfile
import os

from cotk.dataloader import SingleTurnDialog, LanguageGeneration
from cotk._utils import hooks
from cotk._utils.resource_processor import ResourceProcessor

class IWSLT16(SingleTurnDialog):
	'''TODO:
	'''

	ARGUMENTS = SingleTurnDialog.ARGUMENTS
	FILE_ID_DEFAULT = r'''Default: ``resources://OpenSubtitles``.'''
	VALID_VOCAB_TIMES_DEFAULT = r'''Default: ``10``.'''
	MAX_SENT_LENGTH = r'''Default: ``50``.'''
	INVALID_VOCAB_TIMES_DEFAULT = r'''Default: ``0`` (No unknown words).'''
	TOKENIZER_DEFAULT = r'''Default: ``space``'''
	REMAINS_CAPITAL_DEFAULT = r'''Default: ``True``'''
	@hooks.hook_dataloader
	def __init__(self, file_id, min_vocab_times=10, \
			max_sent_length=50, invalid_vocab_times=0, \
			tokenizer="space", remains_capital=True\
			):
		super().__init__(file_id, min_vocab_times, max_sent_length, \
			invalid_vocab_times, tokenizer, remains_capital)

class IWSLT16DEENResourceProcessor(ResourceProcessor):
	'''Processor for IWSLT16 de-en
	'''
	def postprocess(self, local_path):
		'''Postprocess before read.
		'''
		target_path = local_path + "/processed"
		if os.path.isdir(target_path):
			print("Warning: Target dir existed. Skipping IWSLT16DEEN process...")
			return target_path
		os.mkdir(target_path)

		with open(target_path + "/train.txt", 'w', encoding='utf-8') as trainfile:
			with open(local_path + "/train/deen/train.tags.en-de.bpe.de") as postfile:
				with open(local_path + "/train/deen/train.tags.en-de.bpe.en") as respfile:
					for de, en in zip(postfile, respfile):
						trainfile.write(de)
						trainfile.write(en)

		devlist = []
		with open(local_path + "valid.en-de.bpe.de", encoding='utf-8') as postfile:
			with open(local_path + "valid.en-de.bpe.en", encoding='utf-8') as respfile:
				for de, en in zip(postfile, respfile):
					devlist.append(de + en)

		devlist = testlist = devlist

		with open(target_path + "/dev.txt", 'w', encoding='utf-8') as devfile:
			devfile.write("".join(devlist))
		with open(target_path + "/test.txt", 'w', encoding='utf-8') as testfile:
			testfile.write("".join(testlist))

		return target_path


class IWSLT16DEENDistillResourceProcessor(ResourceProcessor):
	'''Processor for IWSLT16 de-en
	'''
	def postprocess(self, local_path):
		'''Postprocess before read.
		'''
		target_path = local_path + "/processed"
		if os.path.isdir(target_path):
			print("Warning: Target dir existed. Skipping IWSLT16DEENDistill process...")
			return target_path
		os.mkdir(target_path)

		with open(target_path + "/train.txt", 'w', encoding='utf-8') as trainfile:
			with open(local_path + "/distill/deen/train.tags.en-de.bpe.de") as postfile:
				with open(local_path + "/distill/deen/train.tags.en-de.bpe.en") as respfile:
					for de, en in zip(postfile, respfile):
						trainfile.write(de)
						trainfile.write(en)

		devlist = []
		with open(local_path + "valid.en-de.bpe.de", encoding='utf-8') as postfile:
			with open(local_path + "valid.en-de.bpe.en", encoding='utf-8') as respfile:
				for de, en in zip(postfile, respfile):
					devlist.append(de + en)

		devlist = testlist = devlist

		with open(target_path + "/dev.txt", 'w', encoding='utf-8') as devfile:
			devfile.write("".join(devlist))
		with open(target_path + "/test.txt", 'w', encoding='utf-8') as testfile:
			testfile.write("".join(testlist))

		return target_path

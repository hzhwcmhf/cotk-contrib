import numpy as np

from cotk.metric import MetricBase

class AspectBasedSentimentAnalysisMetric(MetricBase):
	def __init__(self, dataloader):
		super().__init__()
		self.dataloader = dataloader
		self.label_key = "label"
		self.predict_key = "predict"
		self.predict_force_key = "predict_force"
		self.TP = 0
		self.PN = 0
		self.RN = 0
		self.polarity_right_num = 0
		self.polarity_all_num = 0

	def forward(self, data):
		labels = data[self.label_key]
		predicts = data[self.predict_key]
		predicts_force = data[self.predict_force_key]
		if labels.shape[0] != predicts.shape[0]:
			raise ValueError("Batch num is not matched.")

		self.TP += np.sum(np.logical_and(labels != 0, predicts != 0))
		self.PN += np.sum(predicts != 0)
		self.RN += np.sum(labels != 0)
		self.polarity_right_num += np.sum((predicts_force == labels) * (labels != 0))
		self.polarity_all_num += np.sum(labels != 0)

	def close(self):
		res = {}
		res["aspect_prec"] = self.TP / (self.PN + 1e-10)
		res["aspect_reca"] = self.TP / (self.RN + 1e-10)
		if self.PN == 0:
			res["aspect_prec"] = 0
		if self.RN == 0:
			res["aspect_reca"] = 0
		res["aspect_F1"] = 2 * res["aspect_prec"] * res["aspect_reca"] / \
								(res["aspect_prec"] + res["aspect_reca"] + 1e-10)
		if self.TP == 0:
			res["aspect_F1"] = 0
		res["polarity_acc"] = self.polarity_right_num / (self.polarity_all_num + 1e-10)
		return res

class AspectBasedSentimentAnalysisHardMetric(AspectBasedSentimentAnalysisMetric):
	def __init__(self, dataloader):
		super().__init__(dataloader)
		self.num = 0

	def forward(self, data):
		labels = data[self.label_key]
		predicts = data[self.predict_key]
		predicts_force = data[self.predict_force_key]
		if labels.shape[0] != predicts.shape[0]:
			raise ValueError("Batch num is not matched.")

		hard_mask = np.zeros((labels.shape[0], 1))
		for i, label in enumerate(labels):
			hard_mask[i] = np.sum(np.bincount(label)[1:] != 0) > 1
		self.num += np.sum(hard_mask)
		self.TP += np.sum(np.logical_and(labels != 0, predicts != 0) * hard_mask)
		self.PN += np.sum((predicts != 0) * hard_mask)
		self.RN += np.sum((labels != 0) * hard_mask)
		self.polarity_right_num += np.sum(\
				(np.sum(labels == predicts_force , 1, keepdims=True) == \
				np.sum(labels != 0, 1, keepdims=True)) * hard_mask)
		self.polarity_all_num += np.sum(hard_mask)

	def close(self):
		res = super().close()
		res = {"hard_" + key: value for key, value in res.items()}
		res["hard_num"] = self.num
		return res


class AspectBasedSentimentAnalysisOutofDomainMetric(MetricBase):
	def __init__(self, dataloader):
		super().__init__()
		self.dataloader = dataloader
		self.label_key = "label"
		self.predict_key = "predict"
		self.TP = 0
		self.PN = 0
		self.RN = 0

	def forward(self, data):
		labels = data[self.label_key]
		predicts = data[self.predict_key]
		if labels.shape[0] != predicts.shape[0]:
			raise ValueError("Batch num is not matched.")

		labels_ood = np.sum(labels, 1, keepdims=True) == 0
		predicts_ood = np.sum(predicts, 1, keepdims=True) == 0
		self.TP += np.sum((labels_ood == 0) * (predicts_ood == 0))
		self.PN += np.sum(predicts_ood == 0)
		self.RN += np.sum(labels_ood == 0)

	def close(self):
		res = {}
		res["ood_prec"] = self.TP / (self.PN + 1e-10)
		res["ood_reca"] = self.TP / (self.RN + 1e-10)
		if self.PN == 0:
			res["ood_prec"] = 0
		if self.RN == 0:
			res["ood_reca"] = 0
		res["ood_F1"] = 2 * res["ood_prec"] * res["ood_reca"] / \
								(res["ood_prec"] + res["ood_reca"] + 1e-10)
		return res

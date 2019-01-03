r"""
`contk.dataloader` provides classes and functions downloading and
loading benchmark data automatically. It reduce your cost preprocessing
data and provide a fair dataset for every model. It also help you adapt
your model from one dataset to other datasets.
"""

from .sentiment_analysis import SentimentAnalysis
from .renmin_daily_news import RenminDailyNews

__all__ = ['SentimentAnalysis', 'RenminDailyNews']

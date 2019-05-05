from preprocess import StanfordDocumentTokenizer, TrainTestSplit, Stemming
from lda import LDATraining, LDAPrediction
from regressions import DatasetPrepare, StandardTest

"""
May need to run this:
nltk.download('stopwords')
nltk.download('wordnet')
"""


StanfordDocumentTokenizer.run()
Stemming.run()
TrainTestSplit.run()
LDATraining.run()
LDAPrediction.run()
DatasetPrepare.run()
StandardTest.run()

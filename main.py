from preprocess import StanfordDocumentTokenizer, TrainTestSplit, Stemming
from lda import LDATraining

"""
May need to run this:
nltk.download('stopwords')
nltk.download('wordnet')
"""


StanfordDocumentTokenizer.run()
Stemming.run()
TrainTestSplit.run()
LDATraining.run()

from preprocess import StanfordDocumentTokenizer, TrainTestSplit, Stemming

"""
May need to run this:
nltk.download('stopwords')
nltk.download('wordnet')
"""


StanfordDocumentTokenizer.run()
TrainTestSplit.run()
Stemming.run()

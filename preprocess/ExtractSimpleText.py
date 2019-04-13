import csv
import yaml

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

with open("preproc.config.yml", "r") as cfg_file:
    config = yaml.full_load(cfg_file)

# TODO: fix abbreviations like "U.S."
tokenizer = RegexpTokenizer(r'\w+')
_min_word_length = config['text-simplifying']['min-length']
_stopwords = stopwords.words('english')

if config['text-simplifying']['stemming']:
    from nltk.stem.snowball import SnowballStemmer
    word_to_token = SnowballStemmer("english").stem

elif config['text-simplifying']['lemmatizing']:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    def word_to_token(word): return lemmatizer.lemmatize(word.lower())

else:
    def word_to_token(word): return word


def transform_csv(file_path):
    with open(file_path, "r", newline='') as table_file:
        for row in csv.reader(table_file):
            for headline in row[2:]:
                print(transform_string(headline))
            break


def transform_string(string):
    return list(filter(word_filter_predicate, map(word_to_token, tokenizer.tokenize(string))))


def word_filter_predicate(word):
    return (len(word) >= _min_word_length) and (word not in _stopwords)


if __name__ == '__main__':
    transform_csv("../data/intermediate/train_data_0.csv")

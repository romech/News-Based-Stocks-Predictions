from utils.config import Config

from nltk.corpus import stopwords

# TODO: squash stuff like Russia Today, United States...
config = Config.open("config.yml")
_min_word_length = config("text-simplifying.min-length")
_stopwords = stopwords.words('english')

# At first, choosing between stemming and lemmatizing
if config("text-simplifying.stemming"):
    from nltk.stem.snowball import SnowballStemmer
    word_to_token = SnowballStemmer("english").stem
elif config("text-simplifying.lemmatizing"):
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()
    def word_to_token(word): return _lemmatizer.lemmatize(word.lower())
else:
    def word_to_token(word): return word


def transform_csv(file_in, file_out):
    with open(file_in, "r") as tokens_file, open(file_out, "w") as stems_file:
        for row in tokens_file:
            print(*transform_string(row.strip()), file=stems_file)


def transform_string(string):
    return list(filter(word_filter_predicate, map(word_to_token, string.split(' '))))


def word_filter_predicate(word):
    return word not in _stopwords


def run():
    for i in range(len(config("splits"))):
        transform_csv(config("path.tokens-train").format(i), config("path.stems-train").format(i))
        transform_csv(config("path.tokens-test").format(i), config("path.stems-test").format(i))


if __name__ == '__main__':
    transform_csv(config("path.tokens-train").format(0))

from functools import reduce, lru_cache as cached
import json

import numpy as np
import pandas as pd

from utils.config import Config

config = Config.open("../config.yml")
path = config("path")


@cached()
def load_fasttext_word_vectors():
    word_emb = {}
    with open(config("path.fasttext-model").replace(".bin", ".vec"), "r") as wv:
        wv.readline()  # first line contains number of words and dim
        for line in wv:
            word, vector_str = line.split(" ", 1)
            word_emb[word] = np.fromstring(vector_str, sep=" ")
    return word_emb


def create_visualization_files():
    """Formatted for projector.tensorflow.org"""
    word_emb = load_fasttext_word_vectors()
    np.savetxt("word_vectors.tsv", np.vstack(word_emb.values()), fmt="%.10f", delimiter="\t")
    with open("word_vectors_annotation.tsv", "w") as f:
        for w in word_emb.keys():
            print(w, file=f)


@cached()
def get_word_weights():
    from collections import defaultdict

    words = load_fasttext_word_vectors().keys()
    tf_count, idf_count = learn_tfidf()
    default_value = np.log1p(1) / (1 + np.log1p(0))
    coef = {w: np.log1p(tf_count.get(w, 1)) / (1 + np.log1p(idf_count.get(w, 0))) for w in words}
    return defaultdict(lambda: default_value, coef)


def get_weighted_word_vector(word, vector):
    weights = get_word_weights()
    return vector * weights[word]


def get_weighted_sentence_vector(words, vectors):
    return np.sum([get_weighted_word_vector(w, v) for w, v in zip(words, vectors)], axis=0)


def learn_tfidf():
    import os
    tf_corpus, idf_corpus = path("tf-corpus"), path("idf-corpus")
    if not (os.path.exists(tf_corpus) and os.path.exists(idf_corpus)):
        _make_tfidf_learning_data()

    words = load_fasttext_word_vectors().keys()
    word_set = set(words)
    tf_count = _get_word_counts(tf_corpus, word_set)
    idf_count = _get_word_counts(idf_corpus, word_set)
    return tf_count, idf_count


def _make_tfidf_learning_data():
    from preprocess import StanfordDocumentTokenizer
    from utils.miscellaneous import get_tempfile_name

    with open("data/source/News_Category_Dataset_v2.json", "r") as f:
        articles = [json.loads(line) for line in f]
    categorized_news = pd.DataFrame(articles)
    possibly_related_cats = ["THE WORLDPOST", "POLITICS", "WORLD NEWS"]
    does_matter = reduce(lambda lhs, rhs: lhs | rhs,
                         map(lambda cat: categorized_news.category == cat,
                             possibly_related_cats))

    tf_corp, idf_corp = get_tempfile_name(), get_tempfile_name()
    categorized_news[does_matter]. \
        short_description. \
        to_csv(tf_corp, header=False, index=False)

    categorized_news[~does_matter]. \
        short_description. \
        to_csv(idf_corp, header=False, index=False)

    StanfordDocumentTokenizer.tokenize_file(tf_corp, path("tf-corpus"))
    StanfordDocumentTokenizer.tokenize_file(idf_corp, path("idf-corpus"))


def _get_word_counts(file, vocabulary: set):
    from collections import Counter
    cnt = Counter()
    with open(file, "r") as f:
        for line in f:
            cnt.update(filter(vocabulary.__contains__, line.split()))
    return cnt

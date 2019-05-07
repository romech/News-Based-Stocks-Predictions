import os

import numpy as np
import pandas as pd

from embeddings import word_vectors
from preprocess import StanfordDocumentTokenizer
from utils.config import Config
from utils.tables import matrix_to_tsv
from utils.miscellaneous import get_tempfile_name

config = Config.open("../config.yml")
paths = config("path")


def _prepare_for_fasttext(input_path, output_path):
    if input_path.endswith(".tsv"):
        articles_only = get_tempfile_name()
        pd.read_csv(input_path, sep="\t").sort_values(by="Date").News.to_csv(articles_only, index=False, header=False)
        input_path = articles_only

    StanfordDocumentTokenizer.tokenize_file(input_path, output_path)


def train_fasttext(input_path, output_path):
    os.makedirs(output_path.rsplit("/", 1)[0], exist_ok=True)
    _execute(_fasttext_cmd("skipgram -input {} -output {} -epoch 50 -wordNgrams 2 -lr 0.1 -thread 8".
                           format(input_path, output_path.replace(".bin", ""))))
    os.remove(paths("word-embeds"))  # etc


def make_article_embeddings(model_path, input_path, output_path):
    _execute(_fasttext_cmd("print-sentence-vectors {} < {} > {}".
                           format(model_path, input_path, output_path)))


def make_weighted_article_embeddings(model_path, input_path, output_path):
    wv_file = paths("word-embeds")
    if not os.path.exists(wv_file):  # Creates 1Gb file, takes some time
        _execute(_fasttext_cmd("print-word-vectors {} < {} > {}".
                               format(model_path, input_path, wv_file)))
    with open(input_path, "r") as articles_file,\
            open(wv_file, "r") as vectors_file,\
            open(output_path, "w") as out_file:

        for article in articles_file:
            article_vectors = []
            words = article.strip().split(" ")
            if len(words) == 1 and words[0] == "":
                continue
            for article_word, word_vec in zip(words, vectors_file):
                word, vector_str = word_vec.split(" ", 1)
                if word != article_word:
                    raise Exception("Found mismatch! Expected {}, but found {} {}".
                                    format(article_word, word, vector_str))
                article_vectors.append(np.fromstring(vector_str, sep=" "))
            article_vector = word_vectors.get_weighted_sentence_vector(words, article_vectors)
            print(*article_vector.tolist(), sep="\t", file=out_file)


def _execute(cmd):
    print('Executing', cmd)
    return_code = os.system(cmd)
    if return_code != 0:
        raise Exception('FastText returned error code', return_code)


def _fasttext_cmd(action):
    return "fastText-0.2.0/fasttext " + action


def run_train():
    _prepare_for_fasttext(paths("fasttext-train"), "fasttext-train.txt")
    train_fasttext("fasttext-train.txt", paths("fasttext-model"))


def run():
    make_weighted_article_embeddings(paths("fasttext-model"),
                                     paths("prepare-fasttext"),
                                     paths("article-weighted-embeds"))


if __name__ == '__main__':
    run_train()
    _prepare_for_fasttext(paths("news"), paths("prepare-fasttext"))
    fasttext_out_path = get_tempfile_name()
    make_article_embeddings(paths("fasttext-model"), paths("prepare-fasttext"), fasttext_out_path)
    matrix_to_tsv(fasttext_out_path, paths("article-embeds"))

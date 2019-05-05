import os
import random
import tempfile

import pandas as pd

from preprocess import StanfordDocumentTokenizer
from utils.config import Config
from utils.tables import matrix_to_tsv

config = Config.open("../config.yml")
paths = config("path")


def _prepare_for_fasttext(input_path, output_path):
    if input_path.endswith(".tsv"):
        articles_only = _get_tempfile_name()
        pd.read_csv(input_path, sep="\t").sort_values(by="Date").News.to_csv(articles_only, index=False, header=False)
        input_path = articles_only

    StanfordDocumentTokenizer.tokenize_file(input_path, output_path)


def train_fasttext(input_path, output_path):
    os.makedirs(output_path.rsplit("/", 1)[0], exist_ok=True)
    _execute(_fasttext_cmd("skipgram -input {} -output {} -epoch 50 -wordNgrams 2 -lr 0.1 -thread 8".
                           format(input_path, output_path.replace(".bin", ""))))


def make_article_embeddings(model_path, input_path, output_path):
    _execute(_fasttext_cmd("print-sentence-vectors {} < {} > {}".
                           format(model_path, input_path, output_path)))


def _execute(cmd):
    print('Executing', cmd)
    return_code = os.system(cmd)
    if return_code != 0:
        raise Exception('FastText returned error code', return_code)


def _fasttext_cmd(action):
    return "../fastText-0.2.0/fasttext " + action


def _get_tempfile_name():
    return os.path.join(tempfile.gettempdir(), "embeddings-" + format(random.getrandbits(64), 'x'))


def run_train():
    _prepare_for_fasttext(paths("fasttext-train"), "fasttext-train.txt")
    train_fasttext("fasttext-train.txt", paths("fasttext-model"))


if __name__ == '__main__':
    run_train()
    _prepare_for_fasttext(paths("news"), paths("prepare-fasttext"))
    fasttext_out_path = _get_tempfile_name()
    make_article_embeddings(paths("fasttext-model"), paths("prepare-fasttext"), fasttext_out_path)
    matrix_to_tsv(fasttext_out_path, paths("article-embeds"))

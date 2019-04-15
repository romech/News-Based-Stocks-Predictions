import json
from itertools import repeat

from numpy import savetxt
from scipy.sparse import coo_matrix

from lda import DailyTopics
from utils.iterators import flatten
from utils.config import Config

config = Config.open("../config.yml")


def prepare_topics():
    with open(config("path.splits-tags"), "r") as json_file:
        tags_grouped = json.load(json_file)

    DailyTopics.load_state()

    _save_as_table(DailyTopics.get_for_batch(tags_grouped['train']),
                   config("path.context-topics").format('train'))
    _save_as_table(DailyTopics.get_for_batch(tags_grouped['test']),
                   config("path.context-topics").format('test'))


def _save_as_table(corpus_topics: list, path: str):
    topic_ids, probs = zip(*flatten(corpus_topics))
    doc_ids = list(flatten([repeat(i, len(doc_descr)) for i, doc_descr in enumerate(corpus_topics)]))

    np_matrix = coo_matrix((probs, (doc_ids, topic_ids)), shape=(len(corpus_topics), config("lda.topics"))).todense()
    savetxt(path, np_matrix, fmt='%.8f', delimiter=',')


def run():
    prepare_topics()


if __name__ == '__main__':
    run()

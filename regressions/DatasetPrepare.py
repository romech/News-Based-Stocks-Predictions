import json
from itertools import repeat

from numpy import savetxt
import pandas as pd
from scipy.sparse import coo_matrix

from lda import DailyTopics
from utils.iterators import flatten
from utils.config import Config

config = Config.open("../config.yml")
tags_grouped, target = None, None


def _preload_data():
    global tags_grouped, target
    stocks_table = pd.read_csv(config("path.stocks"), index_col='Date')
    target = _get_target_values(stocks_table)
    stocks_data_tags = set(target.index)

    def in_stocks_data(tag):
        return tag in stocks_data_tags

    with open(config("path.splits-tags"), "r") as json_file:
        tags_grouped = json.load(json_file)
    for part_name, tags in tags_grouped.items():
        tags_grouped[part_name] = list(filter(in_stocks_data, tags))


def prepare_topics():
    DailyTopics.load_state()

    for part in tags_grouped.keys():
        _save_topics_as_table(DailyTopics.get_for_batch(tags_grouped[part]),
                              config("path.context-topics").format(part))


def prepare_stocks():
    for part_name, tags in tags_grouped.items():
        part_index = target.index.intersection(pd.Index(tags))
        pd.DataFrame.to_csv(target.loc[part_index],
                            config("path.target").format(part_name),
                            header=False, index=False)


def _get_target_values(table):
    return (table.Close.diff(-1).shift(1) / table.Open).dropna()


def _save_topics_as_table(corpus_topics: list, path: str):
    topic_ids, probs = zip(*flatten(corpus_topics))
    doc_ids = list(flatten([repeat(i, len(doc_descr)) for i, doc_descr in enumerate(corpus_topics)]))

    np_matrix = coo_matrix((probs, (doc_ids, topic_ids)),
                           shape=(len(corpus_topics), config("lda.topics"))).todense()
    savetxt(path, np_matrix, fmt='%.8f', delimiter=',')


def run():
    _preload_data()
    prepare_topics()
    prepare_stocks()


if __name__ == '__main__':
    run()

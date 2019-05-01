import json
from itertools import repeat

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from lda import DailyTopics
from utils.iterators import flatten
from utils.config import Config

config = Config.open("../config.yml")
tags_grouped, target, sentiment_table, feature_names = None, None, None, None


def _preload_data():
    global tags_grouped, target, sentiment_table
    stocks_table = pd.read_csv(config("path.stocks"), index_col='Date')
    sentiment_table = pd.read_csv(config("path.sentiments"), index_col='Date')
    target = _get_target_values(stocks_table)
    stocks_data_tags = set(target.index)

    def in_stocks_data(tag):
        return tag in stocks_data_tags

    with open(config("path.splits-tags"), "r") as json_file:
        tags_grouped = json.load(json_file)
    for part_name, tags in tags_grouped.items():
        tags_grouped[part_name] = list(filter(in_stocks_data, tags))


def prepare_features_stack():
    DailyTopics.load_state()

    for part, tags in tags_grouped.items():
        _save_feature_matrices(config("path.day-features").format(part),
                               _topics_as_table(DailyTopics.get_for_batch(tags)),
                               _sentiment_as_table(tags)
                               )

    _save_feature_names([f"Topic {i + 1}" for i in range(config("lda.topics"))] + list(sentiment_table.columns))


def prepare_stocks():
    for part_name, tags in tags_grouped.items():
        part_index = target.index.intersection(pd.Index(tags))  # rm

        pd.DataFrame.to_csv(target.loc[part_index],
                            config("path.target").format(part_name),
                            header=False, index=False)


def get_feature_names():
    if feature_names is not None:
        return feature_names
    else:
        with open(config("path.features-descr"), "r") as description_json:
            return list(json.load(description_json).values())


def _get_target_values(table):
    prophet_df = pd.read_csv(config("path.fbprophet"), index_col='Date')
    return prophet_df.Predicted - prophet_df.Actual

# def _get_target_values(table):
#     return (table.Close.diff(-1).shift(1) / table.Open).dropna()


def _topics_as_table(corpus_topics: list):
    if not corpus_topics:
        raise Exception("Empty list")
    topic_ids, probs = zip(*flatten(corpus_topics))
    doc_ids = list(flatten([repeat(i, len(doc_descr)) for i, doc_descr in enumerate(corpus_topics)]))

    return coo_matrix((probs, (doc_ids, topic_ids)),
                      shape=(len(corpus_topics), config("lda.topics"))).todense()


def _sentiment_as_table(tags):
    return sentiment_table.loc[pd.Index(tags)].to_numpy()


def _save_feature_matrices(path, *matrices):
    np.savetxt(path,
               np.hstack(matrices),
               fmt='%.8f', delimiter=',')


def _save_feature_names(names_list):
    with open(config("path.features-descr"), "w") as description_json:
        json.dump(dict(enumerate(names_list, 1)), description_json, indent=True)


def run():
    _preload_data()
    prepare_features_stack()
    prepare_stocks()


if __name__ == '__main__':
    run()

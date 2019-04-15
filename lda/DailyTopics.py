from bisect import bisect_right

from utils.config import Config
from utils.iterators import flatten
from lda import LDAPrediction
from preprocess.TrainTestSplit import news_groupby_tag

config = Config.open("../config.yml")
model, dictionary, news, all_tags = None, None, None, None


_days_in_context = config("regression-data.num-days-context")


def get_for_days(tags: list):
    return LDAPrediction.get_topics_of_doc(model, dictionary, _get_news_by_days(tags))


def repr_for_days(tags: list):
    return LDAPrediction.topics_repr_to_string(get_for_days(tags))


def get_news_context_by_day(tag: str):
    return get_for_days(_days_in_context_with(tag))


def repr_news_context_by_day(tag: str):
    return repr_for_days(_days_in_context_with(tag))


def repr_batch(tags: list):
    return '\n'.join(repr_news_context_by_day(tag) for tag in tags)


def _get_news_by_days(tags: list):
    return ' '.join(flatten(news[tag] for tag in tags))


def _get_news_by_day(tag):
    return ' '.join(news[tag])


def _days_in_context_with(tag: str):
    """
    Returns k known previous days including given one

    >>> _days_in_context_with('2010.01.02')
    ['2009.12.31', '2010.01.01', '2010.01.02']
    """
    right = bisect_right(all_tags, tag) - 1
    left = max(0, right - _days_in_context + 1)
    return all_tags[left:right + 1] if right >= 0 else []


def _load_state():
    global model, dictionary, news, all_tags
    model, dictionary = LDAPrediction.load_model_dict()
    news = news_groupby_tag()
    all_tags = sorted(list(news.keys()))


if __name__ == '__main__':
    _load_state()
    print(repr_batch(['2010.06.07', '2010.06.08']))

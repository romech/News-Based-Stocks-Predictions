import numpy as np
import pandas as pd
from textblob import TextBlob

from utils.config import Config

config = Config.open("../config.yml")


def make_articlewise_sentiments():
    news_table = pd.read_csv(config("path.news"), sep='\t', error_bad_lines=False)
    news_table['Polarity'] = news_table.News.apply(_get_polarity)
    news_table[['Polarity', 'Date', 'News']].to_csv(config("path.article-polarity"), sep='\t', index_label='Id')


def make_daily_sentiments():
    news_table = pd.read_csv(config("path.news"), sep='\t', error_bad_lines=False)
    news_table['Polarity'] = news_table.News.apply(_get_polarity)
    percentile_agg_funcs = [percentile_func(q) for q in config("regression-data.sentiment-percentiles")]

    news_table.groupby('Date')\
        .agg({"Polarity": percentile_agg_funcs})\
        .to_csv(config("path.sentiments"), header=[f.__name__ for f in percentile_agg_funcs])


def _get_polarity(text):
    return TextBlob(text).sentiment.polarity


def percentile_func(value):
    func = lambda series: np.percentile(series, q=value)
    func.__name__ = f"{value}_percentile"
    return func


def run():
    make_daily_sentiments()


if __name__ == '__main__':
    make_articlewise_sentiments()

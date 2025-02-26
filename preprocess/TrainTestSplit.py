from collections import defaultdict
from itertools import islice, repeat
import json

from utils.config import Config
from utils.iterators import flatten

config = Config.open("../config.yml")

"""
Splitting text file assuming it has a form of
HEADER
tag1 token_list
tag1 token_list
...

Resulting files are subsets by tags
"""


def run():
    news = news_groupby_tag()

    tag_list = sorted(list(news.keys()))
    split_tags = {name: make_split(tag_list, bounds[0], bounds[1])
                  for name, bounds in config('splits').dict_like.items()}

    for split_name, tags in split_tags.items():
        news_flat_list = flatten([news[tag] for tag in tags])
        tags_flat_list = flatten([repeat(tag, len(news[tag])) for tag in tags])

        _write_csv(news_flat_list, path=config('path.stems-split').format(split_name))
        _write_csv(tags_flat_list, path=config('path.split-tags').format(split_name))

    with open(config("path.splits-tags"), "w") as json_file:
        json.dump(split_tags, json_file, indent=True)


def news_groupby_tag():
    with open(config('path.stemmed'), "r", newline='') as table_file:
        news = defaultdict(list)
        for line in islice(table_file, 2, None):
            tag_article = line.split(' ', maxsplit=1)
            if len(tag_article) != 2:
                print('Broken line found:', line)
            else:
                news[tag_article[0]].append(tag_article[1].strip())

    return news


def make_split(data, percent_start, percent_end):
    slice_start = len(data) * percent_start // 100
    slice_end = len(data) * percent_end // 100
    return data[slice_start:slice_end]


def _write_csv(rows, path):
    with open(path, 'w') as f:
        for row in rows:
            print(row, file=f)


if __name__ == '__main__':
    run()

import csv
from utils.config import Config

config = Config.open("config.yml")


def run():
    with open(config('path.tokenized'), "r", newline='') as table_file:
        news = list(csv.reader(table_file))[1:]

    for i, split in enumerate(config('splits')):
        train_data, test_data = make_split(news, split['train'], split['test'])

        _write_csv(train_data, config('path.tokens-train').format(i))
        _write_csv(test_data, config('path.tokens-test').format(i))


def make_split(data, train, test):
    train_start, train_end = _slice(len(data), train[0], train[1])
    test_start, test_end = _slice(len(data), test[0], test[1])
    return data[train_start:train_end], data[test_start:test_end]


def _slice(size, start, end):
    return size * start // 100, size * end // 100


def _write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    run()

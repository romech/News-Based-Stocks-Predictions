import csv
from utils.config import Config


config = Config.open("preproc.config.yml")


def run():
    with open(config('source.combined'), "r", newline='') as table_file:
        news = list(csv.reader(table_file))[1:]

    for i, split in enumerate(config('splits')):
        train_data, test_data = make_split(news, split['train'], split['test'])

        _write_csv(train_data, 'train_data_' + str(i))
        _write_csv(test_data, 'test_data_' + str(i))


def make_split(data, train, test):
    train_start, train_end = _slice(len(data), train[0], train[1])
    test_start, test_end = _slice(len(data), test[0], test[1])
    return data[train_start:train_end], data[test_start:test_end]


def _slice(size, start, end):
    return size * start // 100, size * end // 100


def _allocate_output_csv(name):
    return open(config('output_folder') + name + '.csv', 'w', newline='')


def _write_csv(rows, name):
    with _allocate_output_csv(name) as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    run()

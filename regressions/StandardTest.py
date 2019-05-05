from functools import lru_cache as cached

import numpy as np

from regressions import DatasetPrepare, online
from utils.config import Config

config = Config.open("../config.yml")


@cached()
def _load_X(part):
    return np.loadtxt(config("path.day-features").format(part), delimiter=',')


@cached()
def _load_y(part, binary=False):
    target = np.loadtxt(config("path.target").format(part))
    return target > 0 if binary else target


def _describe_features(model):
    feature_names = DatasetPrepare.get_feature_names()
    if hasattr(model, 'feature_importances_'):
        n_features = len(model.feature_importances_)
        top = reversed(sorted(list(zip(model.feature_importances_, range(n_features))))[-10:])
    print('Most important features:', *[f"\n{feature_names[feature]} {prob * 100:.2f}%" for prob, feature in top])


def test(model, description):
    train_X = _load_X('train')
    train_y = _load_y('train')
    test_X = _load_X('test')
    test_y = _load_y('test')

    print('Fitting', description)
    model.fit(train_X, train_y)
    print(f'Scoring {description} on test')
    model.score_online(test_X, test_y)
    _describe_features(model)


def run():
    test(online.RandomForest(), 'Random forest')
    test(online.Lasso(), 'Lasso')


if __name__ == '__main__':
    run()

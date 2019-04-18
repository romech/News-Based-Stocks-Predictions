import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils.config import Config

config = Config.open("../config.yml")

rf = RandomForestClassifier(n_estimators=config("random-forest.n-estimators"))


def train():
    X = _load_X('train')
    y = _load_y('train') > 0
    rf.fit(X, y)


def score():
    X = _load_X('test')
    y = _load_y('test') > 0
    return rf.score(X[:100], y[:100])


def _load_X(part):
    return np.loadtxt(config("path.day-features").format(part), delimiter=',')


def _load_y(part):
    return np.loadtxt(config("path.target").format(part))


def run():
    train()
    print('Most important topics:', sorted(list(zip(rf.feature_importances_, range(100))))[:-10:-1])
    print('Binary classification score (over 100 days):', score())


if __name__ == '__main__':
    run()
    # TODO: write results into file

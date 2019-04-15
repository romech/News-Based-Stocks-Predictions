import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils.config import Config

config = Config.open("../config.yml")

rf = RandomForestClassifier(n_estimators=300)


def train():
    X = _load_X('train')
    y = _load_y('train') > 0
    rf.fit(X, y)


def score():
    X = _load_X('test')
    y = _load_y('test') > 0
    return rf.score(X[:100], y[:100])


def _load_X(part):
    return np.loadtxt(config("path.context-topics").format(part), delimiter=',')


def _load_y(part):
    return np.loadtxt(config("path.target").format(part))


if __name__ == '__main__':
    train()
    print('Binary classification score:', score())

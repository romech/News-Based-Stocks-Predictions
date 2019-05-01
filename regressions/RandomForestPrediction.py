import numpy as np
from sklearn.ensemble import RandomForestClassifier

from regressions import DatasetPrepare
from utils.config import Config

config = Config.open("../config.yml")

rf = RandomForestClassifier(n_estimators=config("random-forest.n-estimators"), warm_start=True, class_weight={False:1, True:1})

# Basically you can do this
# def train():
#     X = _load_X('train')
#     y = _load_y('train') > 0
#     rf.fit(X, y)
#
#
# def score():
#     X = _load_X('test')
#     y = _load_y('test') > 0
#     return rf.score(X[:100], y[:100])


def _load_X(part):
    return np.loadtxt(config("path.day-features").format(part), delimiter=',')


def _load_y(part):
    return np.loadtxt(config("path.target").format(part)) > 0


def score_online():
    batch = config("random-forest.update-every")
    additional_est = config("random-forest.add-estimators")

    train_X = _load_X('train')
    train_y = _load_y('train')
    test_X = _load_X('test')
    test_y = _load_y('test')

    rf.fit(train_X, train_y)
    scores = []
    # re-train the model, test on equally sized batches
    for batch_X, batch_y in zip(np.array_split(test_X, test_X.shape[0] // batch),
                                np.array_split(test_y, test_y.shape[0] // batch)):
        scores.append(rf.score(batch_X, batch_y))
        print('Accuracy:', scores[-1])
        rf.n_estimators += additional_est
        rf.fit(batch_X, batch_y)

    print(f'Accuracy mean: {np.mean(scores):.2f}, std: {np.std(scores):.2f}')


def describe_features():
    feature_names = DatasetPrepare.get_feature_names()
    top = reversed(sorted(list(zip(rf.feature_importances_, range(rf.n_features_))))[-10:])
    print('Most important features:', *[f"\n{feature_names[feature]} {prob * 100:.2f}%" for prob, feature in top])


if __name__ == '__main__':
    score_online()
    describe_features()
    # TODO: write results into file


# Not in use yet
class RandomForestServer:
    """Standalone version"""
    n_est = config("random-forest.n-estimators")
    batch = config("random-forest.update-every")
    additional_est = config("random-forest.add-estimators")

    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=RandomForestServer.n_est, warm_start=True)

    def train(self, features, target):
        self.rf.fit(features, target)

    def score(self, features, target):
        self.rf.score(features, target)

    def score_and_update(self, features, target):
        scores = []
        # re-train the model, test on equally sized batches
        for batch_X, batch_y in zip(np.array_split(features, features.shape[0] // RandomForestServer.batch),
                                    np.array_split(target, target.shape[0] // RandomForestServer.batch)):
            scores.append(rf.score(batch_X, batch_y))
            # print('Accuracy:', scores[-1])
            self.rf.n_estimators += RandomForestServer.additional_est
            self.rf.fit(batch_X, batch_y)

        print(f'Accuracy mean: {np.mean(scores):.2f}, std: {np.std(scores):.2f}')
        return np.mean(scores)

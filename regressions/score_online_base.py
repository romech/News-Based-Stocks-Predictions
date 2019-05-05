import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class OnlineScorer(BaseEstimator):
    """
    Extension for BaseEstimator for models that update upon new data arrival.
    `update` method needs to be override.
    """
    def __init__(self, batch_size=14):
        self.batch_size = batch_size

    def score_online(self, X, y):
        predictions = []
        splits = zip(np.array_split(X, X.shape[0] // self.batch_size),
                     np.array_split(y, y.shape[0] // self.batch_size))
        for batch_X, batch_y in splits:
            predictions.append(self.predict(batch_X))
            self.update(batch_X, batch_y)

        errors = np.concatenate(predictions) - y
        print(pd.Series(errors).describe())
        return np.mean(errors)

    def update(self, X, y):
        raise Exception("Should be implemented if warm start mode is on.")


class FakeOnlineScorer:
    def __init__(self, model, batch_size=14):
        self._params = model.get_params()
        self.model = model
        self.batch_size = batch_size
        self._train_X = self._train_y = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self._train_X = X
        self._train_y = y

    def score_online(self, X, y):
        predictions = []
        splits = zip(np.array_split(X, X.shape[0] // self.batch_size),
                     np.array_split(y, y.shape[0] // self.batch_size))
        for batch_X, batch_y in splits:
            predictions.append(self.model.predict(batch_X))
            self.fit(X=np.vstack([self._train_X, batch_X]),
                     y=np.concatenate([self._train_y, batch_y]))

        errors = np.concatenate(predictions) - y
        print('Absolute error statistics:\n', pd.Series(errors).describe())
        return np.mean(errors)

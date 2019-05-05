import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

from regressions.score_online_base import OnlineScorer, FakeOnlineScorer
from utils.config import Config

config = Config.open("../config.yml")


class RandomForest(RandomForestRegressor, OnlineScorer):
    """Standalone version"""
    n_est = config("random-forest.n-estimators")
    batch = config("random-forest.update-every")
    additional_est = config("random-forest.add-estimators")

    def __init__(self):
        RandomForestRegressor.__init__(self, n_estimators=RandomForest.n_est, warm_start=True)
        OnlineScorer.__init__(self, batch_size=RandomForest.batch)

    def update(self, X, y):
        self.n_estimators += RandomForest.additional_est
        self.fit(X, y)


class Lasso(FakeOnlineScorer):
    def __init__(self):
        super().__init__(model=linear_model.Lasso(alpha=6e-6),
                         batch_size=7)

    @property
    def feature_importances_(self):
        return np.abs(self.model.coef_)

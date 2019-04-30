# Research in progress

import numpy as np
import pandas as pd
from sklearn.cluster.hierarchical import AgglomerativeClustering

from utils.config import Config

config = Config.open("../config.yml")

if __name__ == '__main__':
    df = pd.read_csv(config("path.news"), sep='\t')
    vectors = np.loadtxt(config("path.article-embeds"))
    # checking vectors length: mean ~0.46, std ~ 0.05, max = 1, min ~ 0.29
    # stats = np.array([np.linalg.norm(vectors[i, :]) for i in range(vectors.shape[0])])

    model = AgglomerativeClustering(n_clusters=500)

    pick_recent = 20*365
    clusters = model.fit_predict(vectors[:pick_recent])

    clust_series = pd.Series(clusters, name='Cluster', index=range(pick_recent))

    tagged = pd.concat([clust_series, df.iloc[:pick_recent]], axis=1, sort=False)
    tagged.sort_values(by='Cluster').to_csv("explore_topics.tsv", sep="\t", index=False)

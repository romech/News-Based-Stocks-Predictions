# Research in progress

import numpy as np
import pandas as pd
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from utils.config import Config

config = Config.open("../config.yml")


def cluster_and_dump(model, label):
    df = pd.read_csv(config("path.article-polarity"), sep='\t')
    vectors = np.loadtxt(config("path.article-embeds"))
    # checking vectors length: mean ~0.46, std ~ 0.05, max = 1, min ~ 0.29
    # stats = np.array([np.linalg.norm(vectors[i, :]) for i in range(vectors.shape[0])])

    pick_recent = 20*365
    clusters = model.fit_predict(vectors[:pick_recent])

    clust_series = pd.Series(clusters, name='Cluster', index=range(pick_recent))

    tagged = pd.concat([clust_series, df.iloc[:pick_recent]], axis=1, sort=False)
    tagged.sort_values(by='Cluster').to_csv("explore_topics({}).tsv".format(label), sep="\t", index=False)
    print(tagged.Cluster.value_counts())


if __name__ == '__main__':
    # cluster_and_dump(AgglomerativeClustering(n_clusters=100, affinity="cosine", linkage="ward"), "agglomerative")
    cluster_and_dump(DBSCAN(n_clusters=100, affinity="cosine", linkage="ward"), "dbscan")

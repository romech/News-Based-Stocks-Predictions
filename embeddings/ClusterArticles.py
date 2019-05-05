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

    pick_recent = 3*25*365
    clusters = model.fit_predict(vectors[-pick_recent:])

    clust_series = pd.Series(clusters, name='Cluster')

    tagged = pd.concat([clust_series, df.iloc[-pick_recent:].reset_index(drop=True)], axis=1, sort=False)
    tagged[["News", "Cluster"]].iloc[-pick_recent:].to_csv("for_visualization.tsv", sep="\t", index=False, header=False)
    tagged.sort_values(by='Cluster').to_csv("explore_topics({}).tsv".format(label), sep="\t", index=False)
    print(tagged.Cluster.value_counts())


if __name__ == '__main__':
    cluster_and_dump(AgglomerativeClustering(n_clusters=100, affinity="cosine", linkage="complete"), "100,cos,complete")
    # cluster_and_dump(AgglomerativeClustering(n_clusters=100), "ward")
    # cluster_and_dump(DBSCAN(eps=0.19), "dbscan")

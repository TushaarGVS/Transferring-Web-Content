from __future__ import division

from sklearn import manifold
from sklearn import ensemble
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import forest_cluster as rfc
from forest_cluster import KMedoids
import numpy as np
# from mysuper.datasets import fetch_cars, fetch_10kdiabetes
from scipy.spatial.distance import hamming
import seaborn as sns
import pandas as pd
import time
import scipy.sparse as sp
from classify import get_X_from_elements
from page_data import WebPage

"""
Random Forest clustering works as follows
1. Construct a dissimilarity measure using RF
2. Use an embedding algorithm (MDS, TSNE) to embed into a 2D space preserving that dissimilarity measure.
3. Cluster using K-means or K-medoids
"""

# [1, 2, 3] == 1 , [1, 0, 0]
# [1, 3, 1] == 1, [1, 0, 1]
def fast_hamming_binary_dense(X):
    # does a conversion to dense....
    n_features = X.shape[1]
    D = np.dot(1 - X, X.T)
    return (D + D.T) / X.shape[1]


def fast_hamming_binary_sparse(X, n_matches=None):
    if n_matches:
        n_features = n_matches
    else:
        n_features = X.shape[1]
    H = (X * X.T).toarray()
    return 1 - H / n_features


def fast_hamming_dense(X):
    unique_values = np.unique(X)
    U = sp.csr_matrix((X == unique_values[0]).astype(np.int32))
    H = (U * U.transpose()).toarray()
    for unique_value in unique_values[1:]:
        U = sp.csr_matrix((X == unique_value).astype(np.int32))
        H += (U * U.transpose()).toarray()
    return 1 - H.astype(np.float64) / X.shape[1]

# X = fetch_cars().values
# X = fetch_10kdiabetes(one_hot=False).values
# X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]

elem_list = WebPage.build_from_url("http://www.cs.cmu.edu/~scandal/nesl.html").elementList
X = get_X_from_elements(elem_list)
print(X)

n_trees = 5000
print('\nTree Embedding')
t0 = time.time()
rf = rfc.RandomForestEmbedding(n_estimators=n_trees, random_state=10, n_jobs=-1, sparse_output=False)
leaves = rf.fit_transform(X)
print('Time: %r s', time.time() - t0)


print('Embedding Data')
t0 = time.time()

#if leaves.shape[1] > 50:
#    projection = TruncatedSVD(n_components=50, random_state=123).fit_transform(leaves)
#else:
#    projection = leaves.toarray()
#dissimilarity = fast_hamming_binary_sparse(leaves, n_matches=n_trees)
#projector = manifold.TSNE(random_state=1234, metric='precomputed')
projector = manifold.TSNE(random_state=1234, metric='hamming')
embedding = projector.fit_transform(leaves)


# projector = manifold.MDS(random_state=1234, dissimilarity='precomputed')
# embedding = projector.fit_transform(dissimilarity)
print('Time: %r s', time.time() - t0)


print('Clustering')
t0 = time.time()
clusterer = KMeans(n_clusters=4, random_state=1234, n_init=20, n_jobs=-1)
clusterer.fit(embedding)

# clusterer = KMedoids(n_clusters=3, random_state=1234, distance_metric='precomputed')
# clusterer.fit(np.load('hamming.npy'))
print('Time: %r s', time.time() - t0)


df = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], 'z': clusterer.labels_})
print(df)
# sns.lmplot('x', 'y', hue='z', data=df, fit_reg=False)
print('Done')

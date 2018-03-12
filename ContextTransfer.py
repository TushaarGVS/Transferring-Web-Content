from __future__ import division

# coding: utf-8

# In[10]:

# get_ipython().magic(u'matplotlib inline')

from sklearn import manifold
from sklearn import ensemble
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import forest_cluster as rfc
from forest_cluster import KMedoids
import numpy as np
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import scipy.sparse as sp
from classify import get_X_from_elements
from page_data import WebPage


# In[18]:

def distance(x1, y1, x2, y2):
    return (x1-x2)**2 + (y1-y2)**2

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


# In[40]:


# http://www.google.com
# https://www.ubuntu.com/
# http://www.cs.cmu.edu/~scandal/nesl.html
# http://www.cs.cmu.edu/~scandal/nesl/tutorial2.html

num_clusters = 6

urlsource = 'http://nitk.ac.in/'
urltarget = 'http://nitk.ac.in'

def print_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print '\n'.join(table)


# In[41]:


elem_list = WebPage.build_from_url(urlsource).elementList
X, elem_list = get_X_from_elements(elem_list)

print_matrix(X)


# In[51]:


n_trees = 5000
print('\nTree Embedding')
t0 = time.time()
rf = rfc.RandomForestEmbedding(n_estimators=n_trees, random_state=10, n_jobs=-1, sparse_output=False)
leaves = rf.fit_transform(X)
print('Time: ' + str(time.time() - t0))


print('\nEmbedding Data')
t0 = time.time()

# if leaves.shape[1] > 50:
#     projection = TruncatedSVD(n_components=50, random_state=123).fit_transform(leaves)
# else:
#     projection = leaves.toarray()
# dissimilarity = fast_hamming_binary_sparse(leaves, n_matches=n_trees)
# projector = manifold.TSNE(random_state=1234, metric='precomputed')
projector = manifold.TSNE(random_state=1234, metric='hamming')
embedding = projector.fit_transform(leaves)


# projector = manifold.MDS(random_state=1234, dissimilarity='precomputed')
# embedding = projector.fit_transform(dissimilarity)
print('Time: ' + str(time.time() - t0))


print('\nClustering')
t0 = time.time()
clusterer = KMeans(n_clusters=num_clusters, random_state=1234, n_init=20, n_jobs=-1)
clusterer.fit(embedding)

centroids1 = clusterer.cluster_centers_
print('Cluster Centers:')
print_matrix(centroids1)

# clusterer = KMedoids(n_clusters=num_clusters, random_state=1234, distance_metric='precomputed')
# clusterer.fit(np.load('hamming.npy'))
print('Time: ' + str(time.time() - t0) + '\n')


df = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], 'z': clusterer.labels_})
print(df)
sns.lmplot('x', 'y', hue='z', data=df, fit_reg=False)
plt.show()
print('\nDone {}'.format(len(X)))
print(len(elem_list))

for i in xrange(len(elem_list)):
    elem_list[i].label = clusterer.labels_[i]
    elem_list[i].cluster_x = embedding[:,0][i]
    elem_list[i].cluster_y = embedding[:,1][i]
    print(i,elem_list[i].label)



elem_list2 = WebPage.build_from_url(urltarget).elementList
X, elem_list2 = get_X_from_elements(elem_list2)

print_matrix(X)


# In[51]:


n_trees = 5000
print('\nTree Embedding')
t0 = time.time()
rf = rfc.RandomForestEmbedding(n_estimators=n_trees, random_state=10, n_jobs=-1, sparse_output=False)
leaves = rf.fit_transform(X)
print('Time: ' + str(time.time() - t0))


print('\nEmbedding Data')
t0 = time.time()

# if leaves.shape[1] > 50:
#     projection = TruncatedSVD(n_components=50, random_state=123).fit_transform(leaves)
# else:
#     projection = leaves.toarray()
# dissimilarity = fast_hamming_binary_sparse(leaves, n_matches=n_trees)
# projector = manifold.TSNE(random_state=1234, metric='precomputed')
projector = manifold.TSNE(random_state=1234, metric='hamming')
embedding = projector.fit_transform(leaves)


# projector = manifold.MDS(random_state=1234, dissimilarity='precomputed')
# embedding = projector.fit_transform(dissimilarity)
print('Time: ' + str(time.time() - t0))


print('\nClustering')
t0 = time.time()
clusterer = KMeans(n_clusters=num_clusters, random_state=1234, n_init=20, n_jobs=-1)
clusterer.fit(embedding)

centroids2 = clusterer.cluster_centers_
print('Cluster Centers:')
print_matrix(centroids2)

# clusterer = KMedoids(n_clusters=num_clusters, random_state=1234, distance_metric='precomputed')
# clusterer.fit(np.load('hamming.npy'))
print('Time: ' + str(time.time() - t0) + '\n')


df = pd.DataFrame({'x': embedding[:, 0], 'y': embedding[:, 1], 'z': clusterer.labels_})
print(df)
plt.clf()
sns.lmplot('x', 'y', hue='z', data=df, fit_reg=False)
plt.show()
print('\nDone {}'.format(len(X)))
print(len(elem_list2))

for i in xrange(len(elem_list2)):
    elem_list2[i].label = clusterer.labels_[i]
    elem_list2[i].cluster_x = embedding[:,0][i]
    elem_list2[i].cluster_y = embedding[:,1][i]
    print(i,elem_list2[i].label)

leaf1 = []
leaf2 = []

for e in elem_list2:
    if e.childrenNumber == 0 or (e.tag != 'div' and e.tag!='span'):
        leaf2.append(e)

for e in elem_list:
    if e.childrenNumber == 0 or (e.tag != 'div' and e.tag!='span'):
        leaf1.append(e)
 

print "length of leaf1: {}".format(len(leaf1))
print "length of leaf2: {}".format(len(leaf2))

_map = {}

for e in leaf1:
    print e.tag


    
for e in leaf1:
    cluster_num = e.label
    if not (e.containText or e.containImage):
        continue ;
    dcentroid = distance(e.cluster_x, e.cluster_y, centroids1[cluster_num][0],
                         centroids1[cluster_num][1])
    mind = 10000000
    print(dcentroid)

    for e2 in leaf2:
        if e2.label == cluster_num:
            dcentroid2 = distance(e2.cluster_x, e2.cluster_y,
                                  centroids2[cluster_num][0],
                                  centroids2[cluster_num][1])
            print(dcentroid2)
            dist = abs(dcentroid2 - dcentroid)
            if e2.tag != e.tag:
                dist = dist + 5
            if dist <= mind:
                _map[e] = e2
                mind = dcentroid2


for key, value in _map.items():
    print '{} --> {}'.format(key.tag, value.tag)

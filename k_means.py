import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=300, centers=4, cluster_std=2, random_state=0)
plt.scatter(X[:,0], X[:,1])

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, c='red')
print(kmeans.cluster_centers_)

def Euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def Manhattan_distance(point1, point2):
    return np.sum(np.absolute(point1 - point2))

for i in range(0, len(kmeans.cluster_centers_)):
    print("Manhattan: "+str(Manhattan_distance(kmeans.cluster_centers_[0], kmeans.cluster_centers_[i])))

for i in range(0, len(kmeans.cluster_centers_)):
    print("Euclidean: "+str(Euclidean_distance(kmeans.cluster_centers_[0], kmeans.cluster_centers_[i])))




plt.show()


def cost_per_point(point, center, cost_type = 'E'):
    if cost_type =='E':
        return Euclidean_distance(point, center)**2
    else:
        return Manhattan_distance(point, center)
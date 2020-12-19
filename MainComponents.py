import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# load make_blobs to simulate data
from sklearn.datasets import make_blobs
# load decomposition to do PCA analysis with sklearn
from sklearn import decomposition
X1, Y1 = make_blobs(n_features=20,
         n_samples=1000,
         centers=5, random_state=20,
         cluster_std=5,)

pca = decomposition.PCA(n_components=10)
pc = pca.fit_transform(X1)
columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
pc_df = pd.DataFrame(data = pc, columns = columns)
pc_df['Cluster'] = Y1

df = pd.DataFrame({'var': pca.explained_variance_ratio_,'PC':columns})
print("Матрица факторных нагрузок.")
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(df)

summ = 0
for i in pca.explained_variance_ratio_:
    summ+=i
print(summ)

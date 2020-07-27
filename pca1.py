import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas.plotting import scatter_matrix
dataset = pd.read_csv('data/FinalResultsTable.csv')

#Drop the first column
dataset.drop(dataset.iloc[:, 0:1], inplace=True, axis=1)

#Transpose the data so that the names of the metals are on the Y Axis
#datasetT = dataset.transpose()

#print(datasetT)

X = dataset.iloc[:, 0:8].values
Y = dataset.iloc[:0]

dataset = StandardScaler().fit_transform(dataset)
df = pd.DataFrame(dataset, columns=[Y])
#scatter_matrix(df, alpha=0.2, figsize=(8, 8), diagonal='kde')
#plt.show()

pca = PCA(n_components=2)
pca_model=pca.fit(df)
df_trans=pd.DataFrame(pca_model.transform(df), columns=['pca1', 'pca2'])
rng = np.random.RandomState(0)
colors = rng.rand(18)
sizes = 1000* rng.rand(18)
scatter_proj = plt.scatter(df_trans['pca1'], df_trans['pca2'], c=colors, s=sizes,alpha=0.8)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar()
plt.show()

print(df.iloc[:, 0:8].describe())
#print(Y)

#print(dataset)
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


pca = PCA()
pca_model=pca.fit(df)
plt.plot(np.cumsum(pca_model.explained_variance_ratio_))
plt.axhline(y=0.8, color='r', linestyle='--', linewidth=1)
plt.xlabel('Prinicipal Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
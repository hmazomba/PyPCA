import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


dataset = pd.read_csv('data/FinalResultsTable.csv')

#Drop the first column
dataset.drop(dataset.iloc[:, 0:1], inplace=True, axis=1)

#Transpose the data so that the names of the metals are on the Y Axis
#datasetT = dataset.transpose()
X = dataset.iloc[:, 0:8].values
Y = dataset.iloc[:0]

sns.set(style='white')

#sns.pairplot(dataset)
#plt.tight_layout()
#plt.show()

#print(dataset.head())

#Correlation Matrix
N, _ = dataset.shape
scaler = StandardScaler()
Z = scaler.fit_transform(X)

#Correlation estimation
R = np.dot(Z.T, Z) / N
print(dataset.shape[0], N)

#Eigendecomposition of Correlation Matrix
eigen_values, eigen_vectors = np.linalg.eig(R)

total_var = sum(np.abs(eigen_values))
var_explained =[(i/total_var) for i in sorted(np.abs(eigen_values), reverse=True)]
cum_var_explained=np.cumsum(var_explained)

""" plt.bar(range(1, eigen_values.size + 1), var_explained)
plt.ylabel('Explained variance Ratio')
plt.xlabel('Principal Components')
plt.show() """

value_idx = eigen_values.argsort()[::-1]
eigen_vectors_sorted = eigen_vectors[:, value_idx]

#adding new dimension with np.newaxis
M = np.hstack((eigen_vectors_sorted[0][:, np.newaxis],
               eigen_vectors_sorted[1][:, np.newaxis]))

project_data = Z * M

projected_data = np.asmatrix(Z) * np.asmatrix(M)

colors= ['r', 'b', 'g']
for label, color in zip(np.unique(Y.values), colors):
    idx = dataset[Y].index.values.astype(int)
    x_axis_values = projected_data[idx, 0]
    y_axis_values = projected_data[idx, 1]
    plt.scatter([x_axis_values], [y_axis_values], c=color)



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
sns.pairplot(data=df)
plt.show()
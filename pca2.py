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

pca = PCA()
X_r = pca.fit(X).transform(X)
target_names = dataset.iloc[:0]

def pca_scatter(pca1, pca2):
    plt.close()
    plt.figure()
    colors=['red', 'cyan', 'blue']
    lw = 8
    
    for color, target_name in zip(colors, target_names):
        plt.scatter(X_r[Y==target_name, pca1], X_r[Y==target_name, pca2], color=color, alpha=0.8, lw=lw,
                    label=target_name)
    
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Metal Concentration: Components{} and {}'.format(pca1+1, pca2+1))
    plt.xlabel('Component{}'.format(pca1+1))
    plt.ylabel('Component{}'.format(pca2+1))
    plt.show()
    

pca_scatter(7,8)        
print(target_names)

import GenerateData as data
import matplotlib.pyplot as plt  # python package to plot the graph
import numpy as np
from pandas.tools.plotting import scatter_matrix
from scipy import stats
from sklearn.decomposition import PCA

df=data.readInputfile("./temp.csv")

def scatterPlot(col_num,df):
    plt.scatter(df.index, df.ix[:, col_num], color='r', marker='*', alpha=.4)
    plt.show();

def scatterPlotMatrix():
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

def removeOutliers():
    dfnO=df[(np.abs(stats.zscore(df)) < 1.8).all(axis=1)]
    #del dfnO['Unnamed: 0']
    dfnO=dfnO.reset_index(drop=True)
    dfnO.to_csv("./data.csv",header=None)
    return dfnO

def pca():
    pca=PCA(n_components=3)
    pca.fit(df)
    print(pca.explained_variance_ratio_)


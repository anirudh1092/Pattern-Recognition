import pandas as pd;#python package to read and write the data to csv files
import  numpy as np;#python package to perform mathematical operations
import random


def genRandomList(num_attr,low,high):
    sampl = np.random.uniform(low, high, size=(int(num_attr),))
    return sampl

#This function is used to genrate the Gaussian based data based on user defined values of mu and sigma
def genrateDataVal(mu,sigma,num_attr,num_rows,num_clusters):
    df={}
    frames=[]
    for i in range(0,num_clusters):
        df[i] = pd.DataFrame(data=np.random.normal(mu, sigma, size=(num_rows[i], num_attr)), columns=list(range(num_attr)))
        df[i].loc[:, num_attr+1] = pd.Series(np.ones(num_rows[i])*i)
        frames.append(df[i])
    result=pd.concat(frames).reset_index()
    del result['index']
    result = result.sample(frac=1).reset_index(drop=True)
    result.to_csv("./temp.csv",header=None)
    return result


#This function is used to read the datafile
def readInputfile(fileName):
     dataframe=pd.read_csv(fileName)
     return dataframe


def splitData(df,splitratio):
    df1=df.sample(frac=splitratio)
    df1.to_csv('trainSet.csv')
    df2=df[~df.isin(df1)].dropna()
    df2.drop(df2.columns[len(df2.columns) - 1], axis=1, inplace=True)
    df2.to_csv('testSet.csv')







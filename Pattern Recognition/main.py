import GenerateData as data
import PreprocessData as pre


def main():
    print("Welcome to Pattern Recognition")
    num_cluster=int(input("Enter Number of CLusters of data:  "))
    num_attr=int(input("Enter Number of attributes of the data:  "))
    num_rows = [int(x) for x in input("Enter the number of rows of each Cluster with a 3space :  ").split()]
    mu=data.genRandomList(num_attr,0,10)
    sigma=data.genRandomList(num_attr,0,10)
    df=data.genrateDataVal(mu,sigma,num_attr,num_rows,num_cluster)
    data.splitData(df,0.5)
    pre.removeOutliers()
    pre.pca()# Print the Variance Ratio of top   3 attributes of the PCA


if __name__ == '__main__':
   main();






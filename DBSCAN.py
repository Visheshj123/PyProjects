import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



#------------create datasets--------------------------------

def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    #create random data points (X) and associate the with a cluster (y)
    X,y = make_blobs(n_samples=numSamples, centers = centroidLocation, cluster_std = clusterDeviation)

    #Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(X)
    return X, y

X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)

#----------Modeling----------------------------------------------------------------

radius = 0.3
minimumSamples = 7 #number of samples needed to classify as a core
db = DBSCAN(eps=radius, min_samples=minimumSamples).fit(X)
labels = db.labels_
print(labels) #prints which cluster each data point belongs to

#Distinguish outliers

core_samples_mask = np.zeros_like(db.labels_, dtype=bool) #array of size db.labels with boolean values
core_samples_mask[db.core_sample_indices_] = True #set all core datapoints and borderpoints to True
print(core_samples_mask)

#Number of clusters in labels, for some reason we are negating the -1 cluster, must be some type of noise
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters_)

unique_labels = set(labels) #prints all labels, including outliers

#-------------Data Visualiation----------------------------------------------------------------

# Create colors for each cluster
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))



#plot points with colors

#zip concatentates both value into one object
for k, col in zip(unique_labels, colors):
    if k == -1:
        #means point is an outliers
        col = 'k'
    class_member_mask = (labels == k)  #nparray of all labels that are equal to k

    #plot data points that are clustered together by only taking the datapoints that are True for core_samples_mask
    #and True for the class_member_mask
    #This way, with each iteration it will plot one cluster per iteration
    xy = X[class_member_mask & core_samples_mask]

    #plot clustered data points
    plt.scatter(xy[:,0], xy[:,1], s=50, c=[col], marker = u'o', alpha = 0.5)

    #plot the outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)



plt.savefig('DBSCAN.png')
print(labels)
print(class_member_mask)
print(X[class_member_mask & core_samples_mask])

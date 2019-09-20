import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs


#Generating own data, X1 holds the actual data points of all samples, y1 holds each data points associted cluster
X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)
plt.scatter(X1[:,0], X1[:,1], marker = 'o')
plt.savefig('scatter.png')

#--------------------------------------------agglomerative clustering--------------------------------

#for some reason this ask for the number of clusters to be formed
agglom = AgglomerativeClustering(n_clusters=4, linkage= 'average')
agglom.fit(X1, y1)

plt.figure(figsize = (6,4))

#get min and max range of values
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# Get the average distance between min and max for X1
X1 = (X1 - x_min) / (x_max - x_min)



# This loop displays all of the datapoints.
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]), #places the cluster # at data points, colors them
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.), #gets proper label so that each data point in a cluster is the same color
             fontdict={'weight': 'bold', 'size': 9})




# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.savefig('finalplot.png')




#---------------Dendrogram/phylogentic tree--------------------------------


#create a distance matrix between every point
dist_matrix = distance_matrix(X1, X1)

#define type of heierarchy
Z = hierarchy.linkage(dist_matrix, 'complete')

#display dendrogram
dendro = hierarchy.dendrogram(Z)
plt.savefig('dendo.png')

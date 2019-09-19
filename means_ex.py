import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd

csv_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv"
cust_df = pd.read_csv(csv_path)

print(cust_df.head())


#------------------------------------------Preprocessing--------------------------------

#drop address attribute as it as a catagorical variable
df = cust_df.drop('Address', axis=1)

#Normalize so that all values are weighted equally

from sklearn.preprocessing import StandardScaler
X = df.values[:,1:] #creates array of all values on cust_df starting with column 1 (age)
X = np.nan_to_num(X) #replaces all NaN with ML0101ENv3
Clus_dataSet = StandardScaler().fit_transform(X) #does the standardization and puts into nparray
#print(type(Clus_dataSet))


#-----------------------------------------Modeling--------------------------------

clusternum = 3
k_means = KMeans(init = 'k-means++', n_clusters = clusternum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_ #array that shows which cluster each data point is assigned to

df['Clus_km'] = labels #added labels to each row so we know which cluster each belongs to

#--------------------------------------------Insights--------------------------------

#check centroid values by taking mean of each attribute grouped by cluster
print(df.groupby('Clus_km').mean())
print(X[:, 0])

area = np.pi * (X[:,1]) **2 #taking area of everyone row in attribtue 'education', correlates to the size of the data point cirlce
plt.scatter(X[:,0], X[:, 3], s = area, c=labels.astype(np.float), alpha = 0.5) #alpha is for level of transparenccy, c is for color sequence
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.savefig('k_means.png')





#------------------------------Further Insights------------------------------------------------

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6)) #stroes previos figure in fig object
plt.clf() #clears current figs canvas so that a new fig may be drawn on pyplot
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134) #elev = elevation viewing angle, azim is another viewing angle, rect = size of graph
plt.cla() #clears axis so you can relabel the x,y,z axes
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
plt.savefig('Axes3D.png')

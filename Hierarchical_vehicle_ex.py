import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs
import pylab
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import fcluster


csv_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cars_clus.csv"

#sets include price in thousands (price), engine size (engine_s),
#horsepower (horsepow), wheelbase (wheelbas), width (width), length (length),
#curb weight (curb_wgt), fuel capacity (fuel_cap) and fuel efficiency (mpg).
pdf = pd.read_csv(csv_path)
print(pdf.head())

#For simplicity we drop any attributes with 1+ NULL values, double brackets mean df
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce') #sets invalid enteries to NaN

pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True) #drops index and does NOT add it as a column to the df
print ("Shape of dataset after cleaning: ", pdf.size)

#Dependent Variables
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

#Normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler() #scales each value to betwen 0-1
feature_mtx = min_max_scaler.fit_transform(x) #new weighted feature matrix/numpy array


#----------------------Clustering------------------------------------------------

dist_matrix = distance_matrix(feature_mtx,feature_mtx)
Z = hierarchy.linkage(dist_matrix, 'complete')


agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)
pdf['cluster_'] = agglom.labels_







#print dendo
import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25)
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.savefig('plot.png')



#-------Analysis--------------------------------


print(pdf.groupby(['cluster_','type'])['cluster_'].count())

#Takes mean of all data points in each type of each attribute
agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()


plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k') #labels each datapoint
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label)) #label= is used for the graphs key
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.savefig('plot2.png')

#Test for learning and implementing KNN

#import neccessary tools
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing


#download data set
csv_path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv'
df = pd.read_csv(csv_path)
df.head()


#counts each repeat instance of values in custcat
df['custcat'].value_counts()

#converts DF to numpy array so we can apply scikit-learn, left out custcat
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]


y = df['custcat'].values
y[0:5]

#Normalizing/standardize data to make it more easily comparable
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

#Train, Test, Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

#Classification
from sklearn.neighbors import KNeighborsClassifier

k = 4 #gets 4 nearest neighbors

#generates model based on traning data
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
print(neigh)

yhat = neigh.predict(X_test)
print(yhat[0:5])

#Accuracy Evaluation using Jaccard Index
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Train set Accuracy: ", metrics.accuracy_score(y_test, yhat))

#Plotting Accuracy for different k-values
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

ConfusionMx = [];

for n in range (1, Ks):
    #train model and predict
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
print(mean_acc)

print('highest accuracy obtained was ', mean_acc.max(), 'with a k value of ', mean_acc.argmax()+1)

plt.plot(range(1,Ks),mean_acc,'g')
plt.savefig('KNN.png')

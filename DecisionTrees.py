#creates and produces a decision tree the classifies which drug a patient should take based on their age, sex, cholesterol, age, BP
#Outputs tree visual called DrugTree.png, 

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

csv_path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv'
my_data = pd.read_csv(csv_path)
#print(my_data.head())

#creates new nparray without Drug attriute
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
#converts catagorical variables to numerical variables, needed for decision trees
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder() #auto converts catagotical values to numerical variables
le_sex.fit(['M', 'F']) #applies label encoder to 'M' and 'F' and makes them to 0 and 1
X[:,1] = le_sex.transform(X[:,1]) #applies new labels onto numpy array on second column


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

#print(X[0:5])

#dependent Variable
y = my_data['Drug']
#print(y.head())

#Setting up decision Tree
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X,y, test_size = 0.3, random_state = 3)

#Modeling Decision Tree
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset, y_trainset)

#prediction
predTree = drugTree.predict(X_testset)
print(predTree[0:5])
print(y_testset[0:5])

#Evaluation
from sklearn import metrics
import matplotlib.pyplot as plt

print("Decision Tree Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#Tree visualization
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "DrugTree.png"
feature_names = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=feature_names, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) #use dot data, which is the string representation of the tree and forms a graph
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')

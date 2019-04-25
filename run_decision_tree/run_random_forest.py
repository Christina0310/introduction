from sklearn.ensemble import RandomForestClassifier#BaggingClassifier slow

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("dataset.csv")

#print(dataset.head())

target = dataset.iloc[:, 30].values

#print(target[1:40])

data = dataset.iloc[:,0:30]

#
data_training, data_test, target_training, target_test = train_test_split(data, target, test_size= 0.2, random_state =1 )

#print(data_training.head())
#print(data_test.head())

random_forest_machine = RandomForestClassifier(n_estimators = 11)#default for n is 100, slow

random_forest_machine.fit (data_training, target_training)

predictions = random_forest_machine.predict(data_test)

print(accuracy_score(target_test, predictions))

confusion_matrix = pd.DataFrame(
	confusion_matrix(target_test, predictions),
	columns = ['Predict 0','Predict 1', 'Predict 2', 'Predict 3' ],
	index  = ['True 0', 'True 1', 'True 2', 'True 3'],
	) #correctness and misses, find out too many prediction of 0

print(confusion_matrix)

print(dict(zip(data.columns, decision_tree_machine.feature_importances_)))
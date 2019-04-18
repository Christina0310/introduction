import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix#measure how confused the machine is 


dataset = pd.read_csv("dataset.csv")

print(dataset.head())

target = dataset.iloc[:, 30].values

print(target[1:40])

data = dataset.iloc[:,0:30]

print(data.head())

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size= 0.2, random_state =1)

print(data_training.head())
print(data_test.head())

decision_tree_machine = tree.DecisionTreeClassifier(criterion = "gini", max_depth =10)# gini impurity for entropy measure; or "entropy"
decision_tree_machine.fit(data_training, target_training)#random state for tree as well but not that differ

predictions = decision_tree_machine.predict(data_test)

print(accuracy_score(target_test, predictions))#compare target_test and prediction

confusion_matrix = pd.DataFrame(
	confusion_matrix(target_test, predictions),
	columns = ['Predict 0','Predict 1', 'Predict 2', 'Predict 3' ],
	index  = ['True 0', 'True 1', 'True 2', 'True 3'],
	) #correctness and misses, find out too many prediction of 0

print(confusion_matrix)

print(dict(zip(data.columns, decision_tree_machine.feature_importances_)))#compare importance within;zip: print things pretty; dictionary
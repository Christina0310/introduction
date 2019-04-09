import pandas as pd #linear regression
from sklearn.model_selection import train_test_split#_underscore: library-module;split: split data hold out method
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold

dataset = pd.read_csv("dataset.csv")#default is true: title row

#print(dataset.head())

data = dataset.iloc[:,3:10] #extract subset of data , right limit not included
#or change into [3:8]
#print(data.head())

target = dataset.iloc[:,2].values# break the value or it not work : unfloat

#print(target)#unfloat no head

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size = 0.25, random_state = 0)#randomstate: same

print(data_training.head())
print(data_test.head())
print(target_training)#target no head.
print(target_test)#test 25% controlc terminate

print(data.shape)#dimension practice short key: every line the same
print(target.shape)
print(data_test.shape)
print(target_training.shape)
print(target_test.shape)

linear_machine = linear_model.LinearRegression()#keep the machine
linear_machine.fit (data_training, target_training)
prediction = linear_machine.predict(data_test)#2500 result

print(prediction)

plt.scatter(target_test, prediction)
plt.xlabel('target test')
plt.ylabel('prediction')

#plt.savefig('scatter_test_prediction.png')

#print(metrics.r2_score(target_test,prediction))

data = data.values

#print(data)

kfold_machine = KFold(n_splits =4)
kfold_machine.get_n_splits(data)
print(kfold_machine)

for training_index, test_index in kfold_machine.split(data):
    print("Training:", training_index)
    print("Test:", test_index)#rest of it is training dataset
    data_training, data_test = data[training_index], data[test_index]#array of array
    target_training, target_test = target[training_index], target[test_index]
    linear_machine = linear_model.LinearRegression()
    linear_machine.fit (data_training, target_training)
    prediction = linear_machine.predict(data_test)
    print(metrics.r2_score(target_test,prediction))











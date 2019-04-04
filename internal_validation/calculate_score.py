import pandas as pd #linear regression
from sklearn.model_selection import train_test_split#_underscore: library-module;split: split data

dataset = pd.read_csv("dataset.csv")#default is true: title row

#print(dataset.head())

data = dataset.iloc[:,3:10] #extract subset of data , right limit not included

#print(data.head())

target = dataset.iloc[:,2].values# break the value or it not work : unfloat

#print(target)#unfloat no head

data_train, data_test, target_training, target_test = train_test_split(data, target, test_size = 0.25, random_state = 0)#randomstate: same

print(data_train.head())
print(data_test.head())
print(target_training)#target no head.
print(target_test)#target 25% controlc terminate

print(data.shape)#dimension
print(target.shape)
print(data_test.shape)
print(target_training.shape)
print(target_test.shape)
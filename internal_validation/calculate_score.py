import pandas as pd #linear regression

dataset = pd.read_csv("dataset.csv")#default is true: title row

#print(dataset.head())

data = dataset.iloc[:,3:10] #extract subset of data , right limit not included

#print(data.head())

target = dataset.iloc[:,2].values# break the value or it not work : unfloat

print(target)#unfloat no head
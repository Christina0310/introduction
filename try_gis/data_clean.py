import pandas as pd 
import numpy as np 

crime_data = pd.read_csv("Crimes_-_2001_to_present.csv")                     # load data;read_csv function

'''print(type(crime_data))

print(crime_data.shape)

print(crime_data.tail())

print(crime_data.dtypes)

print(crime_data.describe())    '''                                              # iloc to reassign the dataframe value
###############################################
df = pd.DataFrame(crime_data)

print(df.tail())

df.replace('', np.nan, inplace = True)

df.dropna(inplace=True)

print(df.shape)
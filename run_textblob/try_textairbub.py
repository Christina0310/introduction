from textblob import TextBlob
from nltk.tokenize import TabTokenizer 
from textblob.sentiments import NaiveBayesAnalyzer
import pandas as pd 
import numpy as np 

dataset = pd.read_csv("airbnb-reviews-part")


print(comment)
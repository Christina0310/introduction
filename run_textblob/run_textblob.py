from textblob import TextBlob
from nltk.tokenize import TabTokenizer
from textblob import Word 
from textblob.sentiments import NaiveBayesAnalyzer

'''text_blob_object = TextBlob("This is a Good Try! You are the best!")

print(text_blob_object.tags)# part of speech

tokenizer = TabTokenizer()

text_blob_object = TextBlob("This is a Good Try! You are the best!", tokenizer =tokenizer)
print(text_blob_object.tokens)
print(text_blob_object.words)
print(text_blob_object.sentences)'''

'''word = Word("went")#	Word : constructor
print(word.lemmatize("v"))#sort words by grouping inflected or variant forms of the same word
#verb
text_blob_object = TextBlob("Hhiss is a correct sentence")
print(text_blob_object.correct())'''

text_blob_object = TextBlob("This is a Good Try! You are the best!",analyzer = NaiveBayesAnalyzer())
print(text_blob_object.sentiment)


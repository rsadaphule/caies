

import numpy as np 
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer


nltk.download('all', quiet=True)

data = pd.read_csv('./data/Musical_instruments_reviews.csv')
stem = data["summary"].values


#Word Tokenize
#Lets up up 2nd element from data frame
strInput = stem[2]
wtokens = nltk.word_tokenize(strInput)
print("input string : " +  strInput)
print("word tokens: {}".format(', '.join(wtokens)))

#Social media tokens
ctokens = nltk.casual_tokenize(strInput)
print("social media tokens: {}".format(', '.join(ctokens)))

def get_tokens(sent):
    return nltk.word_tokenize(sent)

def get_stemmed(tokens):
    stemmed = []
    for token in tokens:
        stemmed.append(PorterStemmer().stem(token))
        #stemmed.append(LancasterStemmer().stem(token))
        #stemmed.append(SnowballStemmer('english').stem(token))
    return stemmed

stemmed = get_stemmed(get_tokens(strInput))
print("after stemming: {}".format(', '.join(stemmed)))

from nltk.stem.wordnet import WordNetLemmatizer
lemm = WordNetLemmatizer()

def do_lemma(tokens):
    lemms = []
    for token in tokens:
        lemms.append(lemm.lemmatize(token, 'v'))
    result = dict(zip(tokens, lemms))
    return result

strInput = "I recorded and payed the song very well"
lemmed = do_lemma(get_tokens(strInput))
print("{tokens:lemmas}", lemmed)
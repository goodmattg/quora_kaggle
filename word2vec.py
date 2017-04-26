import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics.pairwise import cosine_similarity

wv_model = Word2Vec(brown.sents())

def read_data(path_to_file):
    df = pd.read_csv(path_to_file)
    # Remove missing values and duplicates from training data
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df

data = read_data("input/train.csv")
X = data[['question1','question2']].values
y = data['is_duplicate'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_sample = X_train[:2]
for x in X_sample:
    print x[0]
    print x[1]
    s1 = [w.lower() for w in tokenizer.tokenize(x[0])]
    s2 = [w.lower() for w in tokenizer.tokenize(x[1])]
    for w1 in s1:
        for w2 in s2:
            try:
                sim_score = wv_model.wv.similarity(w1,w2)
            except KeyError:
                sim_score = 'Not in vocabulary'
            print w1,w2,sim_score

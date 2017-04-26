import pandas as pd
from sklearn.model_selection import train_test_split
from dependency_parse import get_pos_dep
from semantic_similarity import align
from features import percentage_semantic_similarity_both

data = pd.read_csv("input/train.csv")
X = data[['question1','question2']].values
y = data['is_duplicate'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_sample = X_train[:10]
for x in X_sample:
    print x[0]
    print x[1]
    S = get_pos_dep(x[0])
    T = get_pos_dep(x[1])
    A, info_A = align(S,T)
    p = percentage_semantic_similarity_both(S,T,A)
    print 'Percentage Semantic Similarity: ', p
    print '\n'

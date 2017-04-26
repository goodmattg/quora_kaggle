import pandas as pd

data = pd.read_csv("input/train.csv")
X = data[['id','question1','question2','is_duplicate']].values
print X[0]

for x in X:
    id = x[0]
    q1 = x[1]
    q2 = x[2]
    dup = x[3]
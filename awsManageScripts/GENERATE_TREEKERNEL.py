
from pycorenlp import StanfordCoreNLP
import TreeKernel as tk
import TreeBuild as tb
import numpy as np
import pandas as pd
import pickle
import sys

nlp = StanfordCoreNLP('http://localhost:9000')

def _getNLPToks_(rawSentence):
    try:
        output = nlp.annotate(rawSentence, properties={
            'annotators': 'tokenize,ssplit,pos,parse',
            'outputFormat': 'json'
        })
    except UnicodeDecodeError:
        sentence = unidecode(rawSentence)
        output = nlp.annotate(sentence, properties={
            'annotators': 'tokenize,ssplit,pos,parse',
            'outputFormat': 'json'
        })
    tokens = output['sentences'][0]['tokens']
    parse = output['sentences'][0]['parse'].split("\n")
    return {
        'toks':tokens, 'parse':parse
    }


# ARGS:
# [1] Path to data file
# [2] lambda
# [3] SST_ON
# [4] path to output

if __name__ == "__main__":
    feature_vect = {}

    with open(sys.argv[1], 'rb') as handle:
        count = 0
        while True:
            if (count % 10000 == 0):
                print(count)
            try:
                unit = pickle.load(handle)
                # ST Syntax score
                (rscore_st, nscore_st) = tk._CollinsDuffy_(unit['q1']['parse'], unit['q2']['parse'], sys.argv[2], 1, sys.argv[3])

                feature_vect[unit['id']] = {
                    'id':unit['id'],
                    'cdNorm_st':nscore_st
                }

            except EOFError:
                break
            except:
                
                try:
                    print("Quote error on: %d" % unit['id'])

                    q1_stanford = _getNLPToks_(unit['q1']['raw'].replace('"','').replace("'",''))
                    q2_stanford = _getNLPToks_(unit['q2']['raw'].replace('"','').replace("'",''))

                    tree_1 = tb.tree()
                    tree_2 = tb.tree()

                    # Generate a tree structure
                    tb._generateTree_(q1_stanford['parse'], tree_1)
                    tb._generateTree_(q2_stanford['parse'], tree_2)

                    # Flip the trees
                    tb._flipTree_(tree_1)
                    tb._flipTree_(tree_2)

                    (rscore_st, nscore_st) = tk._CollinsDuffy_(tree_1, tree_2, sys.argv[2], 1, sys.argv[3])

                    feature_vect[unit['id']] = {
                        'id':unit['id'],
                        'cdNorm_st':nscore_st
                    }

                    print("Quote error resolved")
                    pass
            
                except:
                    print("Unable to resolve: %d" % unit['id'])
                

            count += 1

    df_feature = pd.DataFrame.from_dict(feature_vect)
    df_feature = df_feature.transpose()

    df_feature[['id']] = df_feature[['id']].astype(int)

    df_feature.to_csv(sys.argv[4], index=False)


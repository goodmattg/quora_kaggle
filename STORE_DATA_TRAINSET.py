import pandas as pd
from pycorenlp import StanfordCoreNLP
from math import ceil
import TreeBuild as tb
import pickle
import json, os
import gzip
import sys

def read_data(path_to_file):
    df = pd.read_csv(path_to_file)
    print ("Shape of base training File = ", df.shape)
    df.dropna(inplace=True)
    print("Shape of base training data after cleaning = ", df.shape)
    return df

def _getNLPToks_(rawSentence):
    try:
        output = nlp.annotate(rawSentence, properties={
            'annotators': 'tokenize,ssplit,pos,parse,ner,depparse',
            'outputFormat': 'json'
        })
    except:
        print("Stanford NLP crash on row")
        return

    if (isinstance(output, str)):
        # output = json.loads(output) # Convert str output to dict
        print("Error processing row. Attempt to strip % and quotes")
        return _getNLPToks_(rawSentence.replace("%","").replace('"','').replace("'",''))

    dependencies = output['sentences'][0]['basicDependencies']
    tokens = output['sentences'][0]['tokens']
    parse = output['sentences'][0]['parse'].split("\n")

    return {'deps':dependencies,
            'toks':tokens,
            'parse':parse}


if __name__ == "__main__":

    nlp = StanfordCoreNLP('http://localhost:9000')

    # First argument is path to input dataframe
    dataframe = pd.read_csv(sys.argv[1])

    block_ctr = 1
    count = 0

    fout = gzip.open(sys.argv[2] + str(block_ctr) + '.nlp', 'wb')

    for row in dataframe.iterrows():
        try:
            q1_stanford = _getNLPToks_(row[1]['question1'])
            q2_stanford = _getNLPToks_(row[1]['question2'])

            tree_1 = tb.tree()
            tree_2 = tb.tree()

            # Generate a tree structure
            tb._generateTree_(q1_stanford['parse'], tree_1)
            tb._generateTree_(q2_stanford['parse'], tree_2)

            # Flip the trees
            tb._flipTree_(tree_1)
            tb._flipTree_(tree_2)

            tmp = {'q1': {
                    'raw': row[1]['question1'],
                    'toks': q1_stanford['toks'],
                    'deps': q1_stanford['deps'],
                    'parse': tree_1
                    },
                   'q2': {
                    'raw': row[1]['question2'],
                    'toks': q2_stanford['toks'],
                    'deps': q2_stanford['deps'],
                    'parse': tree_2
                    },
                   'id':row[1]['id'],
                   'is_duplicate':row[1]['is_duplicate']
                   }

            pickle.dump(tmp, fout, protocol=pickle.HIGHEST_PROTOCOL)

            tree_1.clear()
            tree_2.clear()

        except OSError:
            fout.close()
            block_ctr += 1
            fout = gzip.open(sys.argv[2] + str(block_ctr) + '.nlp', 'wb')
            pickle.dump(tmp, fout, protocol=pickle.HIGHEST_PROTOCOL)
            tree_1.clear()
            tree_2.clear()

        except:
            print("Failure on row: %d" % count)

        count+=1

    print("NLP Tree Generation completed!")



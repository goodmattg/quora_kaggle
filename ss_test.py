import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from dependency_parse import get_pos_dep
from semantic_similarity import align
from features import percentage_semantic_similarity_both

with open('input/stanfordData_train1.nlp', 'rb') as handle:
    count = 0
    rows = []
    while count < 200:
        try:
            d = pickle.load(handle)
            id = d['id']
            is_duplicate = d['is_duplicate']
            #print(d['q1']['raw'])
            #print(d['q2']['raw'])
            S = get_pos_dep(d['q1']['toks'], d['q1']['deps'])
            T = get_pos_dep(d['q2']['toks'], d['q2']['deps'])
            A = align(S,T)
            #print(A)
            #print('Len(S):',len(S))
            #print('Len(T):', len(T))
            #print('Len(A):', len(A))
            # Semantic Similarity Features
            #S_sem_sim = percentage_semantic_similarity_one(S, A)
            #T_sem_sim = percentage_semantic_similarity_one(T, A)
            sem_sim = percentage_semantic_similarity_both(S, T, A)
            # Put all features in a row
            features_row = [id, sem_sim, is_duplicate]
            rows.append(features_row)
            count += 1
            print(d['q1']['raw'])
            print(S)
            print(d['q2']['raw'])
            print(T)
            print('Similarity:', sem_sim)
            print(d['is_duplicate'])
            print('\n')
        except EOFError:
            break

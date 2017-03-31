
# coding: utf-8

# In[ ]:

import numpy as np


# In[2]:

def functionWordWAN(sentence, fwords_dict, window_size, alpha):
    wan = np.zeros((len(fwords_dict), len(fwords_dict)))
    for idx, pivot in enumerate(sentence[:-1]): # don't include last word as pivot
        if (not fwords_dict.get(pivot)):
            continue
        else:
            sentence_slice = sentence[idx:(idx+window_size)]
            for it, word in enumerate(sentence_slice[1:]): # don't include pivot word
                if (fwords_dict.get(word)):
                    if (fwords_dict.get(word) != fwords_dict.get(pivot)): # no self loops
                        r = fwords_dict.get(pivot)
                        c = fwords_dict.get(word)
                        wan[r-1,c-1] += pow(alpha,it+1)
                           
    return wan


# In[1]:

def normalizeWAN(raw_wan, fwords_len):
    sums = raw_wan.sum(axis=1)
    sums = np.matlib.repmat(sums, fwords_len, 1)
    sums = sums.T
    norm_wan = raw_wan / sums
    norm_wan = np.nan_to_num(norm_wan) # Make sure nans from zero division are zeros
    return norm_wan


# In[ ]:




# In[ ]:




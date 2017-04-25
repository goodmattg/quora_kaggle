
# coding: utf-8

# In[7]:

import numpy as np
from numpy import linalg as la
from numpy import inf
import functools


# In[1]:

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


# In[2]:

def normalizeWAN(raw_wan, fwords_len):
    sums = raw_wan.sum(axis=1)
    sums = np.matlib.repmat(sums, fwords_len, 1)
    sums = sums.T
    norm_wan = raw_wan / sums
    norm_wan = np.nan_to_num(norm_wan) # Make sure nans from zero division are zeros
    return norm_wan


# In[3]:

def relativeEntropy(wan1, wan2):
    # Return is a list containing 1 w.r.t 2, then 2 w.r.t. 1
    data = {}
    data['entropies'] = [0.0, 0.0]
    # 50 selected as reasonable convergence value
    limiting1 = la.matrix_power(wan1, 50) 
    limiting2 = la.matrix_power(wan2, 50)
    # Store stationary distributions
    data['l1'] = limiting1
    data['l2'] = limiting2
    # 1 w.r.t 2
    imd = np.nan_to_num(np.divide(wan1,wan2)) # Set all nan's to zero (0/0)
    imd[(imd == inf) | (imd == -inf)] = 0 # All infinities to 0 (sc alar/0)
    log_imd = np.nan_to_num(np.log(imd))
    log_imd[(log_imd == inf) | (log_imd == -inf)] = 0    
    weights = functools.reduce(np.multiply, [limiting1, wan1, log_imd])
    data['entropies'][0] = weights.sum()
    # 2 w.r.t 1
    imd = np.nan_to_num(np.divide(wan2,wan1)) # Set all nan's to zero (0/0)
    imd[(imd == inf) | (imd == -inf)] = 0 # All infinities to 0 (scalar/0)
    log_imd = np.nan_to_num(np.log(imd))
    log_imd[(log_imd == inf) | (log_imd == -inf)] = 0    
    weights = functools.reduce(np.multiply, [limiting2, wan2, log_imd])
    data['entropies'][1] = weights.sum()
    
    return data


# In[4]:

def generateWANPair(snt1, snt2, window_size, alpha):
    ret_data = {}
    snt1_set = set(snt1)
    snt2_set = set(snt2)
    comp_set = snt1_set.union(snt2_set)
    comp_dict = {}
    # Generate the composite sentence dictionary
    for idx, v in enumerate(comp_set):
        comp_dict[v] = idx+1
    
    ret_data['snt1_wan'] = functionWordWAN(snt1, comp_dict, window_size, alpha)
    ret_data['snt2_wan'] = functionWordWAN(snt2, comp_dict, window_size, alpha)
    
    return ret_data
        


# In[ ]:




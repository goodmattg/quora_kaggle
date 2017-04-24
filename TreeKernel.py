
# coding: utf-8

import numpy as np
import math

'''
Helper Methods
'''

def _isLeaf_(tree, parentNode):
    return (len(tree[parentNode]['children']) == 0)

def _isPreterminal_(tree, parentNode):
    for idx in tree[parentNode]['children']:
        if not _isLeaf_(tree, idx):
            return False
    return True

'''
Implementation of the Colins-Duffy or Subset-Tree (SST) Kernel
'''

def _cdHelper_(tree1, tree2, node1, node2, store, lam, SST_ON):
    # No duplicate computations
    if store[node1, node2] >= 0:
        return

    # Leaves yield similarity score by definition
    if (_isLeaf_(tree1, node1) or _isLeaf_(tree2, node2)):
        store[node1, node2] = 0
        return

    # same parent node
    if tree1[node1]['posOrTok'] == tree2[node2]['posOrTok']:
        # same children tokens
        if tree1[node1]['childrenTok'] == tree2[node2]['childrenTok']:
            # Check if both nodes are pre-terminal
            if _isPreterminal_(tree1, node1) and _isPreterminal_(tree2, node2):
                store[node1, node2] = lam
                return
            # Not pre-terminal. Recurse among the children of both token trees.
            else:
                nChildren = len(tree1[node1]['children'])
                runningTotal = 0
                for idx in range(nChildren):
                     # index ->  node_id
                    tmp_n1 = tree1[node1]['children'][idx]
                    tmp_n2 = tree2[node2]['children'][idx]
                    # Recursively run helper
                    _cdHelper_(tree1, tree2, tmp_n1, tmp_n2, store, lam, SST_ON)
                    runningTotal *= (SST_ON + store[tmp_n1, tmp_n2])

                store[node1, node2] = lam * runningTotal
                return
        else:
            store[node1, node2] = 0
    else: # parent nodes are different
        store[node1, node2] = 0
        return

def _cdKernel_(tree1, tree2, lam, SST_ON):
    # Fill the initial state of the store
    store = np.empty((len(tree1), len(tree2)))
    store.fill(-1)
    # O(N^2) to compute the tree dot product
    for i in range(len(tree1)):
        for j in range(len(tree2)):
            _cdHelper_(tree1, tree2, i, j, store, lam, SST_ON)

    return store.sum()

'''
Returns a tuple w/ format: (raw, normalized)
If NORMALIZE_FLAG set to False, tuple[1] = -1
'''
def _CollinsDuffy_(tree1, tree2, lam, NORMALIZE_FLAG, SST_ON):
    raw_score = _cdKernel_(tree1, tree2, lam, SST_ON)
    if (NORMALIZE_FLAG):
        t1_score = _cdKernel_(tree1, tree1, lam, SST_ON)
        t2_score = _cdKernel_(tree2, tree2, lam, SST_ON)
        return (raw_score,(raw_score / math.sqrt(t1_score * t2_score)))
    else:
        return (raw_score,-1)



# In[ ]:

'''
Implementation of the Partial Tree (PT) Kernel from:
"Efficient Convolution Kernels for Dependency and Constituent Syntactic Trees"
by Alessandro Moschitti
'''

'''
The delta function is stolen from the Collins-Duffy kernel
'''

def _deltaP_(tree1, tree2, seq1, seq2, store, lam, mu, p):

#     # Enumerate subsequences of length p+1 for each child set
    if p == 0:
        return 0
    else:
        # generate delta(a,b)
        _delta_(tree1, tree2, seq1[-1], seq2[-1], store, lam, mu)
        if store[seq1[-1], seq2[-1]] == 0:
            return 0
        else:
            runningTot = 0
            for i in range(p-1, len(seq1)-1):
                for r in range(p-1, len(seq2)-1):
                    scaleFactor = pow(lam, len(seq1[:-1])-i+len(seq2[:-1])-r)
                    dp = _deltaP_(tree1, tree2, seq1[:i], seq2[:r], store, lam, mu, p-1)
                    runningTot += (scaleFactor * dp)
            return runningTot

def _delta_(tree1, tree2, node1, node2, store, lam, mu):
#     print("Evaluating Delta: (%d,%d)" % (node1, node2))

    # No duplicate computations
    if store[node1, node2] >= 0:
        return

    # Leaves yield similarity score by definition
    if (_isLeaf_(tree1, node1) or _isLeaf_(tree2, node2)):
        store[node1, node2] = 0
        return

    # same parent node
    if tree1[node1]['posOrTok'] == tree2[node2]['posOrTok']:

        if _isPreterminal_(tree1, node1) and _isPreterminal_(tree2, node2):
            if tree1[node1]['childrenTok'] == tree2[node2]['childrenTok']:
                store[node1, node2] = lam
            else:
                store[node1, node2] = 0
            return

        else:
            # establishes p_max
            childmin = min(len(tree1[node1]['children']), len(tree2[node2]['children']))
            deltaTot = 0
            for p in range(1,childmin+1):
                # compute delta_p
                deltaTot += _deltaP_(tree1, tree2,
                                     tree1[node1]['children'],
                                     tree2[node2]['children'], store, lam, mu, p)

            store[node1, node2] = mu * (pow(lam,2) + deltaTot)
            return

    else:
        # parent nodes are different
        store[node1, node2] = 0
        return

def _ptKernel_(tree1, tree2, lam, mu):
    # Fill the initial state of the store
    store = np.empty((len(tree1), len(tree2)))
    store.fill(-1)

    # O(N^2) to compute the tree dot product
    for i in range(len(tree1)):
        for j in range(len(tree2)):
            _delta_(tree1, tree2, i, j, store, lam, mu)

    return store.sum()

'''
Returns a tuple w/ format: (raw, normalized)
If NORMALIZE_FLAG set to False, tuple[1] = -1
'''
def _MoschittiPT_(tree1, tree2, lam, mu, NORMALIZE_FLAG):
    raw_score = _ptKernel_(tree1, tree2, lam, mu)
    if (NORMALIZE_FLAG):
        t1_score = _ptKernel_(tree1, tree1, lam, mu)
        t2_score = _ptKernel_(tree2, tree2, lam, mu)
        return (raw_score,(raw_score / math.sqrt(t1_score * t2_score)))
    else:
        return (raw_score,-1)



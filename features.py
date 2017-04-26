def percentage_semantic_similarity_one(S,A):
    return float(len(A)) / len(S)

def percentage_semantic_similarity_both(S,T,A):
    return 2.0 * len(A) / (len(S) + len(T))

def percentage_nouns_matched(S,T,A):
    S_nouns = 0
    return


def word2vec_nouns_unmatched(S,T,A):
    # For unmatched noun, penalize it. Compare to other unmatched nouns.
    return


def dif_nouns_unmatched(S,T,A):
    # Return number of nouns
    return
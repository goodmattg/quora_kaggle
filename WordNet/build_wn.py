from collections import defaultdict
from nltk.corpus import wordnet as wn
import json

def get_synonyms(pos):
    # All nouns, adjectives, and verbs
    synsets = list(wn.all_synsets(pos))
    all_synonyms = defaultdict(set)
    for synset in synsets:
        original_lemmas = [str(lemma.name()).replace('_',' ') for lemma in synset.lemmas()]
        for original_lemma in original_lemmas:
            all_synonyms[original_lemma] = all_synonyms[original_lemma].union(set(original_lemmas))
    # Make the sets lists, so it can be turned into JSON
    all_synonyms = {str(k).lower(): list([str(m).lower() for m in v]) for k, v in all_synonyms.items() if list(v)}
    return all_synonyms

def create_json(d, filepath):
    json.dump(d, open(filepath, 'w'))

d_nouns = get_synonyms('n')
d_adjs = get_synonyms('a')
d_verbs = get_synonyms('v')

#create_json(d_nouns, 'wn_synonyms_n.json')
#create_json(d_adjs, 'wn_synonyms_a.json')
#create_json(d_verbs, 'wn_synonyms_v.json')


# Works!
d = json.load(open('wn_synonyms_n.json'))
print d['dog']
d = json.load(open('wn_synonyms_a.json'))
print d['cool']
d = json.load(open('wn_synonyms_v.json'))
print d['run']
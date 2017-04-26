from collections import defaultdict
import json

def parse_ppdb():
    filepath = 'ppdb-1.0-xl-lexical'
    d = defaultdict(set)
    with open(filepath) as f:
        ppdb = f.readlines()

    for line in ppdb:
        l = line.split(' ||| ')
        w1 = l[1]
        w2 = l[2]
        d[w1].add(w2)
        d[w2].add(w1)

    d = {str(k).lower(): list([str(m).lower() for m in v]) for k, v in d.items() if list(v)}
    return d

def create_json(d, filepath):
    json.dump(d, open(filepath, 'w'))

#d = parse_ppdb()
#create_json(d, 'ppdb_xl_paraphrases.json')

# Works!
d = json.load(open('ppdb_xl_paraphrases.json'))
print d['dog']
from pycorenlp import StanfordCoreNLP
from collections import defaultdict
from unidecode import unidecode
# Provides functions to return the pos and dep dicts and length for a sentence.


def get_pos_dep(sentence):
    nlp = StanfordCoreNLP('http://localhost:9000')
    try:
        output = nlp.annotate(sentence, properties={
          'annotators': 'tokenize,ssplit,pos,depparse',
          'outputFormat': 'json'
          })
    except UnicodeDecodeError:
        sentence = unidecode(sentence)
        print sentence
        output = nlp.annotate(sentence, properties={
            'annotators': 'tokenize,ssplit,pos,depparse',
            'outputFormat': 'json'
        })

    dependencies = output['sentences'][0]['basicDependencies']
    # Dictionary of tokenized sentence index values to word, pos, and dependencies.
    S = defaultdict(dict)

    # First, get words and pos tags
    tokens = output['sentences'][0]['tokens']
    for t in tokens:
        i = t['index'] - 1 # Shift the index by 1
        word = str(t['word']).lower()
        S[i]['word'] = word
        S[i]['pos'] = t['pos']
        S[i]['deps'] = {}

    # Then get dependencies
    for dep in dependencies:
        g = dep['governor'] - 1
        if g < 0: # Don't include the ROOT as a governor, for now.
            continue
        d = dep['dependent'] - 1
        S[g]['deps'][d] = dep['dep']

    return S

# Testing
#S = "Whenever I go home, I'm happy to see you"
#d = get_pos_dep(S)
#print d
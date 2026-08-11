import nltk
from nltk import CFG
from nltk.parse import RecursiveDescentParser

grammar = CFG.fromstring("""
S -> NP_S VP_S
S -> NP_P VP_P

NP_S -> Det_S N_S
NP_P -> Det_P N_P

VP_S -> V_S
VP_P -> V_P

Det_S -> 'the'
Det_P -> 'the'

N_S -> 'boy' | 'girl' | 'cat'
N_P -> 'boys' | 'girls' | 'cats'

V_S -> 'runs' | 'eats' | 'plays'
V_P -> 'run' | 'eat' | 'play'
""")

parser = RecursiveDescentParser(grammar)

sentence = input("Enter a sentence: ").lower().split()

trees = list(parser.parse(sentence))

if trees:
    print("Sentence has correct agreement.")
    for tree in trees:
        print(tree)
else:
    print("Sentence has incorrect agreement.")

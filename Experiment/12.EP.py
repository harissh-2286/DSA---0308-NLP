from nltk import CFG
from nltk.parse import EarleyChartParser

grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'the' | 'a'
N -> 'cat' | 'dog'
V -> 'chased' | 'saw'
""")

parser = EarleyChartParser(grammar)

sentence = "the dog saw a cat".split()

for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()

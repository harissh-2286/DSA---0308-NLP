from nltk import CFG
from nltk.parse import ChartParser

grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'the' | 'a'
N -> 'cat' | 'dog'
V -> 'chased' | 'saw'
""")

parser = ChartParser(grammar)

sentence = "the cat chased a dog".split()

for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()

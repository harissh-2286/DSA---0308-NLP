import nltk
from nltk import PCFG
from nltk.parse import ViterbiParser
grammar=PCFG.fromstring("""
            S -> NP VP [1.0]
            NP -> Det N [0.6]
            NP -> 'students' [0.4]
            VP -> V NP [0.5]
            VP -> V [0.5]
            Det -> 'the' [1.0]
            N -> 'students' [1.0]
            V -> 'study' [1.0]
            """)
token="the students study".split()
parser=ViterbiParser(grammar)
trees=list(parser.parse(token))
for tree in trees:
    tree.pretty_print()
    print(tree.prob())

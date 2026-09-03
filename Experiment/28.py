import nltk
from nltk import word_tokenize, pos_tag
from nltk.chunk import RegexpParser
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

sentence = input("Enter a sentence: ")

words = word_tokenize(sentence)

tagged_words = pos_tag(words)

print("\nPOS Tagged Words:")
print(tagged_words)

grammar = r"""
    NP: {<DT>?<JJ.*>*<NN.*>+}
"""

chunk_parser = RegexpParser(grammar)

tree = chunk_parser.parse(tagged_words)

print("\nNoun Phrases:")
noun_phrases = []

for subtree in tree.subtrees():
    if subtree.label() == "NP":
        np_words = [word for word, tag in subtree.leaves()]
        noun_phrase = " ".join(np_words)
        noun_phrases.append(np_words)

        print(noun_phrase)

print("\nSemantic Analysis:")

for np_words in noun_phrases:

    head_word = np_words[-1]

    synsets = wordnet.synsets(head_word)

    if synsets:
        meaning = synsets[0].definition()

        print("\nNoun Phrase:", " ".join(np_words))
        print("Head Word:", head_word)
        print("Meaning:", meaning)

    else:
        print("\nNoun Phrase:", " ".join(np_words))
        print("Head Word:", head_word)
        print("Meaning: Meaning not found")

import nltk
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
words = ["running", "playing", "studies", "studying","connected", "connection", "happiness","flying", "wolves", "cars"]
print("Original Word\t\tStem")
print("-"*40)
for word in words:
    stem=stemmer.stem(word)
    print(f"{word:10}\t\t{stem}")

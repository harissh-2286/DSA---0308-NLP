import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sentence
sentence = "I went to the bank to deposit money"

# Convert sentence into words
words = sentence.split()

# Apply Lesk algorithm to the word "bank"
result = lesk(words, "bank")
print("Sentence:", sentence)
print("Ambiguous word: bank")
if result:
    print("Selected sense:", result.name())
    print("Definition:", result.definition())
else:
    print("No sense found")

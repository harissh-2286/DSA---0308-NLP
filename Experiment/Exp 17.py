import nltk
from nltk.corpus import wordnet

# Download WordNet
nltk.download('wordnet')
nltk.download('omw-1.4')

# Word to explore
word = "car"

# Retrieve synsets
synsets = wordnet.synsets(word)

print("Word:", word)
print("Number of synsets:", len(synsets))

# Display each synset
for synset in synsets:
    print("\nSynset:", synset.name())
    print("Definition:", synset.definition())
    print("Examples:", synset.examples())

    # Display synonyms
    synonyms = synset.lemmas()
    print("Synonyms:", [lemma.name() for lemma in synonyms])

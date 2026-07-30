import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

word = input("Enter a word: ")

print("Stemmed Word:", stemmer.stem(word))
print("Lemmatized Word:", lemmatizer.lemmatize(word))

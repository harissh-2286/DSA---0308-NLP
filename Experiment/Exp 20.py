from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Collection of documents
documents = [
    "Python is a popular programming language",
    "Machine learning uses Python for data analysis",
    "Natural language processing deals with text data",
    "Information retrieval searches and ranks documents",
    "Python is useful for machine learning and data science"
]

# Get search query from user
query = input("Enter your search query: ")

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert documents into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert query into TF-IDF vector
query_vector = vectorizer.transform([query])

# Calculate cosine similarity
similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

# Rank documents according to similarity
ranked_documents = similarity_scores.argsort()[::-1]

# Display results
print("\nSearch Results")
print("=" * 50)

for rank, index in enumerate(ranked_documents, start=1):
    print("Rank:", rank)
    print("Score:", round(similarity_scores[index], 4))
    print("Document:", documents[index])
    print("-" * 50)

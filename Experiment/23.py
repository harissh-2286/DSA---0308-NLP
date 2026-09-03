import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)

def evaluate_coherence(text):
    sentences = nltk.sent_tokenize(text)

    if len(sentences) < 2:
        print("At least two sentences are required.")
        return

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    similarities = []

    for i in range(len(sentences) - 1):
        similarity = cosine_similarity(
            tfidf_matrix[i:i+1],
            tfidf_matrix[i+1:i+2]
        )[0][0]

        similarities.append(similarity)

    coherence_score = sum(similarities) / len(similarities)

    print("TEXT COHERENCE EVALUATION")
    print("=" * 60)

    for i, score in enumerate(similarities):
        print(
            f"Sentence {i+1} -> Sentence {i+2}: "
            f"{score:.2f}"
        )

    print("\nOverall Coherence Score:",
          round(coherence_score, 2))

    if coherence_score >= 0.50:
        result = "Highly Coherent"
    elif coherence_score >= 0.25:
        result = "Moderately Coherent"
    else:
        result = "Poorly Coherent"

    print("Coherence Level:", result)


text = """
Natural language processing helps computers understand human language.
It is widely used in chatbots and virtual assistants.
These applications can answer questions and provide useful information.
Therefore, NLP is an important area of artificial intelligence.
"""

evaluate_coherence(text)

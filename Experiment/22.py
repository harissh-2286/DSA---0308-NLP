import nltk
import re

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

def reference_resolution(text):
    sentences = nltk.sent_tokenize(text)

    noun_references = []

    print("REFERENCE RESOLUTION")
    print("=" * 60)

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)

        resolved_sentence = sentence

        for word, tag in tagged_words:
            if tag.startswith("NN"):
                noun_references.append(word)

        pronouns = ["he", "she", "it", "they", "him", "her", "them"]

        for word in words:
            if word.lower() in pronouns and noun_references:
                reference = noun_references[-1]

                pattern = r"\b" + re.escape(word) + r"\b"
                resolved_sentence = re.sub(
                    pattern,
                    f"{word} ({reference})",
                    resolved_sentence,
                    count=1,
                    flags=re.IGNORECASE
                )

        print("\nOriginal Sentence:")
        print(sentence)

        print("Resolved Sentence:")
        print(resolved_sentence)


text = """
John went to the library. He borrowed a book.
The book was interesting. It contained many useful examples.
Mary met John there. She discussed the book with him.
"""

reference_resolution(text)

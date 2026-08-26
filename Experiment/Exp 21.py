import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Get sentence from user
sentence = input("Enter a sentence: ")

# Process the sentence
doc = nlp(sentence)

print("\nSyntax-Driven Semantic Analysis")
print("-" * 50)

# Extract noun phrases
for chunk in doc.noun_chunks:
    noun_phrase = chunk.text
    head_word = chunk.root.text

    # Simple semantic interpretation
    if head_word.lower() in ["student", "teacher", "doctor", "person", "man", "woman"]:
        meaning = "Person"
    elif head_word.lower() in ["college", "school", "company", "university", "hospital"]:
        meaning = "Organization/Institution"
    elif head_word.lower() in ["book", "laptop", "phone", "car", "computer"]:
        meaning = "Object"
    elif head_word.lower() in ["chennai", "india", "london", "paris"]:
        meaning = "Location"
    else:
        meaning = "General Entity"

    print("Noun Phrase :", noun_phrase)
    print("Head Word   :", head_word)
    print("Meaning     :", meaning)
    print("-" * 50)

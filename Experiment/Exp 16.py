import spacy

# Load the pre-trained English NLP model
nlp = spacy.load("en_core_web_sm")

# Get text from the user
text = input("Enter a text: ")

# Process the text
doc = nlp(text)

# Display named entities
print("\nNamed Entities:")
print("-" * 40)

for ent in doc.ents:
    print("Entity:", ent.text)
    print("Type  :", ent.label_)
    print("Start :", ent.start_char)
    print("End   :", ent.end_char)
    print("-" * 40)

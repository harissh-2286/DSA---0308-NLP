from transformers import pipeline

translator = pipeline(
    "translation_en_to_fr",
    model="Helsinki-NLP/opus-mt-en-fr"
)

english_text = "Hello, how are you? I am learning NLP."

result = translator(english_text)

print("English:", english_text)
print("French:", result[0]["translation_text"])

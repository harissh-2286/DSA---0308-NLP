from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-en-fr"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Artificial intelligence is changing the world."

inputs = tokenizer(text, return_tensors="pt")

translated_tokens = model.generate(**inputs)

translation = tokenizer.decode(
    translated_tokens[0],
    skip_special_tokens=True
)

print("English:", text)
print("French:", translation)

from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

prompt = "Write a short paragraph about Artificial Intelligence."

response = client.responses.create(
    model="gpt-5.6-luna",
    input=prompt
)

print("Prompt:")
print(prompt)

print("\nGenerated Text:")
print(response.output_text)

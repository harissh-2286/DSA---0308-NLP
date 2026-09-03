from google import genai

client = genai.Client(api_key="Your Api Key")
prompt = input("Enter your prompt: ")
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)
print("\nGenerated Text:")
print(response.text)

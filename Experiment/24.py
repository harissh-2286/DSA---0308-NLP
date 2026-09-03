import re

def recognize_dialog_act(sentence):
    text = sentence.lower().strip()

    if re.search(r'\b(hello|hi|hey|good morning|good afternoon)\b', text):
        return "Greeting"

    elif re.search(r'\b(bye|goodbye|see you|good night)\b', text):
        return "Goodbye"

    elif text.endswith('?') or re.search(
        r'\b(what|where|when|who|why|how|can|could|would|do|does|is|are)\b',
        text
    ):
        return "Question"

    elif re.search(
        r'\b(please|could you|can you|would you|give me|help me)\b',
        text
    ):
        return "Request"

    elif re.search(r'\b(thank you|thanks|thank)\b', text):
        return "Thanking"

    elif re.search(r'\b(yes|sure|okay|ok|correct)\b', text):
        return "Affirmation"

    elif re.search(r'\b(no|not|never|wrong)\b', text):
        return "Negation"

    else:
        return "Statement"

dialog = [
    "Hello!",
    "How are you?",
    "I am fine, thank you.",
    "Could you help me with my homework?",
    "Sure, I can help you.",
    "Thank you!",
    "Goodbye!"
]

print("DIALOG ACT RECOGNITION")
print("-" * 35)

for speaker, sentence in enumerate(dialog, start=1):
    act = recognize_dialog_act(sentence)

    print(f"Utterance {speaker}: {sentence}")
    print(f"Dialog Act: {act}\n")

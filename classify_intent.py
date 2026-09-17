import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from intents import INTENTS

# Load the API key from .env so it's never hardcoded in this file
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Build the few-shot part of the prompt once, from intents.py,
# instead of hardcoding examples again in this file
def build_intent_reference():
    lines = []
    for intent_name, info in INTENTS.items():
        lines.append(f"\n{intent_name}: {info['description']}")
        for example in info["examples"]:
            lines.append(f"  example: \"{example}\"")
    return "\n".join(lines)

INTENT_REFERENCE = build_intent_reference()
VALID_INTENTS = list(INTENTS.keys())

def classify_message(customer_message):
    prompt = f"""You are classifying a customer support message from a Twitter conversation with AppleSupport.

Here are the possible intents, each with example messages:
{INTENT_REFERENCE}

Classify this new message into exactly ONE of these intents: {", ".join(VALID_INTENTS)}

Message: "{customer_message}"

Respond with ONLY the intent name, nothing else. No explanation, no punctuation."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
        reasoning_effort="low"
    )

    predicted_intent = response.choices[0].message.content.strip()
    print(f"RAW MODEL OUTPUT: {repr(predicted_intent)}")

    if predicted_intent not in VALID_INTENTS:
        predicted_intent = "general_bug_other"

    return predicted_intent


if __name__ == "__main__":
    # Test run on a small sample first, not the whole 20k rows,
    # since this costs one API call per row
    df = pd.read_csv('data/clean_pairs.csv')
    sample = df.sample(n=20, random_state=1)

    results = []
    for _, row in sample.iterrows():
        message = row['customer_text_clean']
        predicted = classify_message(message)
        results.append({"message": message, "predicted_intent": predicted})
        print(f"[{predicted}] {message[:80]}")

    pd.DataFrame(results).to_csv('data/sample_classified.csv', index=False)
    print("\nSaved to data/sample_classified.csv")
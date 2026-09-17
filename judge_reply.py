import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def judge_reply(customer_message, generated_reply, good_reply_notes):
    prompt = f"""You are evaluating the quality of a customer support reply.

Customer message: "{customer_message}"

What a good reply should cover: {good_reply_notes}

Generated reply to evaluate: "{generated_reply}"

Rate the generated reply on these 4 criteria, each from 1 to 5:
- relevance: does it actually address the customer's specific problem?
- accuracy: is it grounded and correct, not making up information or promises?
- tone: is it appropriately empathetic and professional for a support reply?
- completeness: does it cover what a good reply should, based on the notes above?

Respond with ONLY a JSON object in this exact format, nothing else:
{{"relevance": <1-5>, "accuracy": <1-5>, "tone": <1-5>, "completeness": <1-5>, "brief_reason": "<one short sentence>"}}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
        reasoning_effort="low"
    )

    raw_output = response.choices[0].message.content.strip()

    # The model sometimes wraps JSON in markdown code fences despite
    # instructions not to — strip those if present before parsing
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        raw_output = raw_output.replace("json", "", 1).strip()

    try:
        scores = json.loads(raw_output)
    except json.JSONDecodeError:
        # If parsing fails, return a clearly-flagged failure instead of
        # crashing the whole evaluation run
        scores = {
            "relevance": None, "accuracy": None, "tone": None,
            "completeness": None, "brief_reason": "JSON PARSE FAILED",
            "raw_output": raw_output
        }

    return scores


if __name__ == "__main__":
    test_customer_message = "my phone battery drains so fast since the update"
    test_reply = "We're sorry to hear that! Could you let us know which iOS version you're on (Settings > General > About) and if you've tried any battery-saving tips yet? We'll help get it back to normal."
    test_notes = "acknowledge frustration, ask for iOS version, suggest battery tips, offer further help"

    result = judge_reply(test_customer_message, test_reply, test_notes)
    print(json.dumps(result, indent=2))
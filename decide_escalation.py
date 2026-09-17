from transformers import pipeline

# Twitter-specific sentiment model — trained on real tweets, not generic text,
# so it handles slang/caps/profanity better than a generic sentiment model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Intents that should always go to a human, regardless of sentiment —
# these involve money, policy judgment, or safety, which an auto-reply
# shouldn't attempt to resolve on its own
ALWAYS_ESCALATE_INTENTS = {"billing_issue"}


def get_sentiment(message):
    result = sentiment_analyzer(message[:512])[0]  # model has a token limit
    return result['label'], result['score']


def decide_escalation(customer_message, predicted_intent):
    sentiment_label, sentiment_score = get_sentiment(customer_message)

    # Rule 1: certain intents always escalate, no matter the tone
    if predicted_intent in ALWAYS_ESCALATE_INTENTS:
        return {
            "escalate": True,
            "reason": f"intent '{predicted_intent}' always requires human review (billing/policy judgment)",
            "sentiment": sentiment_label
        }

    # Rule 2: strongly negative sentiment with high confidence suggests
    # a frustrated customer who may need more than a canned reply
    if sentiment_label == "negative" and sentiment_score > 0.95:
        return {
            "escalate": True,
            "reason": f"strongly negative sentiment (confidence {sentiment_score:.2f}) suggests customer frustration beyond standard fix",
            "sentiment": sentiment_label
        }

    # Rule 3: catch-all/unclear intent means we're not confident enough
    # to auto-generate a reliable reply
    if predicted_intent == "general_bug_other":
        return {
            "escalate": True,
            "reason": "intent unclear or doesn't match a known issue type, needs human judgment",
            "sentiment": sentiment_label
        }

    # Otherwise: common, well-understood issue, not strongly negative — safe to auto-handle
    return {
        "escalate": False,
        "reason": f"common '{predicted_intent}' issue with standard fix available, sentiment not severe ({sentiment_label})",
        "sentiment": sentiment_label
    }


if __name__ == "__main__":
    test_cases = [
        ("my phone battery drains so fast since the update", "battery_charging_issue"),
        ("Oi what the fuck have you done to the battery on my iPhone? Since the new update it dies in 2 hours", "battery_charging_issue"),
        ("I was charged twice for my subscription this month", "billing_issue"),
    ]

    for message, intent in test_cases:
        result = decide_escalation(message, intent)
        print(f"\nMessage: {message}")
        print(f"Intent: {intent}")
        print(f"Escalate: {result['escalate']}")
        print(f"Reason: {result['reason']}")
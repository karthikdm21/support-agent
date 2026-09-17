import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

GOLDEN_SET_PATH = "data/golden_set.csv"
OUTPUT_PATH = "data/baseline_trivial_results.csv"

GENERIC_REPLY = "Thanks for reaching out — a specialist will follow up with you shortly."


def to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() in {"true", "1", "yes"}


def main():
    df = pd.read_csv(GOLDEN_SET_PATH)

    # Majority-class intent, computed from the golden set's true labels
    majority_intent = df["true_intent"].value_counts().idxmax()
    print(f"Majority intent in golden set: '{majority_intent}' "
          f"({(df['true_intent'] == majority_intent).sum()} / {len(df)} rows)")

    df["baseline_predicted_intent"] = majority_intent
    df["baseline_reply"] = GENERIC_REPLY
    df["baseline_escalate"] = False  # trivial baseline never escalates

    intent_acc = accuracy_score(df["true_intent"], df["baseline_predicted_intent"])

    y_true_escalate = df["true_escalate"].apply(to_bool)
    y_pred_escalate = df["baseline_escalate"]
    escalate_acc = accuracy_score(y_true_escalate, y_pred_escalate)
    e_precision, e_recall, e_f1, _ = precision_recall_fscore_support(
        y_true_escalate, y_pred_escalate, average="binary", zero_division=0
    )

    print("\n" + "=" * 60)
    print("TRIVIAL BASELINE")
    print("=" * 60)
    print(f"Intent accuracy (always predict '{majority_intent}'): {intent_acc:.3f}")
    print(f"Escalation accuracy (never escalate):                 {escalate_acc:.3f}")
    print(f"Escalation precision:                                 {e_precision:.3f}")
    print(f"Escalation recall:                                    {e_recall:.3f}")
    print(f"Escalation F1:                                        {e_f1:.3f}")
    print("Reply quality: not LLM-judged here — it's the same canned sentence")
    print("for every message, included so the reply-quality judge has a floor")
    print("to compare your system's replies against.")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
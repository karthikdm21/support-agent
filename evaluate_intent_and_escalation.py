import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from decide_escalation import decide_escalation

GOLDEN_SET_PATH = "data/golden_set.csv"
OUTPUT_PATH = "data/eval_results.csv"


def main():
    df = pd.read_csv(GOLDEN_SET_PATH)

    required_cols = {
        "customer_text_clean",
        "predicted_intent",
        "true_intent",
        "true_escalate",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"golden_set.csv is missing expected columns: {missing}")

    # --- Intent accuracy ---
    y_true_intent = df["true_intent"]
    y_pred_intent = df["predicted_intent"]

    intent_accuracy = accuracy_score(y_true_intent, y_pred_intent)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, average="macro", zero_division=0
    )

    print("=" * 60)
    print("INTENT CLASSIFICATION")
    print("=" * 60)
    print(f"Accuracy:            {intent_accuracy:.3f}")
    print(f"Macro Precision:     {precision:.3f}")
    print(f"Macro Recall:        {recall:.3f}")
    print(f"Macro F1:            {f1:.3f}")
    print()
    print("Per-intent breakdown:")
    print(classification_report(y_true_intent, y_pred_intent, zero_division=0))

    # --- Run escalation decision on every golden-set row ---
    print("=" * 60)
    print("Running decide_escalation() on each golden-set row...")
    print("=" * 60)

    predicted_escalate = []
    predicted_reason = []
    for _, row in df.iterrows():
        result = decide_escalation(row["customer_text_clean"], row["predicted_intent"])
        predicted_escalate.append(result["escalate"])
        predicted_reason.append(result["reason"])

    df["predicted_escalate"] = predicted_escalate
    df["predicted_escalate_reason"] = predicted_reason

    # normalize true_escalate to bool in case it's stored as string/int in the CSV
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        return str(val).strip().lower() in {"true", "1", "yes"}

    y_true_escalate = df["true_escalate"].apply(to_bool)
    y_pred_escalate = df["predicted_escalate"].apply(to_bool)

    escalate_accuracy = accuracy_score(y_true_escalate, y_pred_escalate)
    e_precision, e_recall, e_f1, _ = precision_recall_fscore_support(
        y_true_escalate, y_pred_escalate, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true_escalate, y_pred_escalate, labels=[True, False])

    print()
    print("=" * 60)
    print("ESCALATION DECISION")
    print("=" * 60)
    print(f"Accuracy:            {escalate_accuracy:.3f}")
    print(f"Precision (escalate):{e_precision:.3f}")
    print(f"Recall (escalate):   {e_recall:.3f}")
    print(f"F1 (escalate):       {e_f1:.3f}")
    print()
    print("Confusion matrix [rows=true, cols=predicted], labels=[True, False]:")
    print(cm)
    print()

    mismatches = df[y_true_escalate != y_pred_escalate]
    print(f"Escalation mismatches: {len(mismatches)} / {len(df)}")
    if len(mismatches) > 0:
        print("\nSample mismatches (up to 5):")
        cols_to_show = [
            "customer_text_clean",
            "predicted_intent",
            "true_escalate",
            "predicted_escalate",
            "predicted_escalate_reason",
        ]
        print(mismatches[cols_to_show].head(5).to_string(index=False))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFull per-row results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
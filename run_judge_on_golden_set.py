import pandas as pd
from generate_reply import generate_reply
from judge_reply import judge_reply
import time

# Update this path if your file is named differently
golden = pd.read_csv('data/golden_set.csv')

results = []

for i, row in enumerate(golden.itertuples()):
    # First generate a reply for this customer message
    generated = generate_reply(row.customer_text_clean)

    # Then have the judge score that generated reply
    scores = judge_reply(
        row.customer_text_clean,
        generated,
        row.good_reply_notes
    )

    results.append({
        "customer_text": row.customer_text_clean,
        "generated_reply": generated,
        "relevance": scores.get("relevance"),
        "accuracy": scores.get("accuracy"),
        "tone": scores.get("tone"),
        "completeness": scores.get("completeness"),
        "judge_reason": scores.get("brief_reason"),
    })

    if i % 10 == 0:
        print(f"Judged {i}/{len(golden)}")

    time.sleep(2.0)  # two LLM calls per row now (generate + judge), stay under rate limit

results_df = pd.DataFrame(results)
results_df.to_csv('data/judged_replies.csv', index=False)

print("\n--- AVERAGE SCORES ---")
for criterion in ["relevance", "accuracy", "tone", "completeness"]:
    avg = results_df[criterion].mean()
    print(f"{criterion}: {avg:.2f} / 5")

print(f"\nSaved detailed results to data/judged_replies.csv")
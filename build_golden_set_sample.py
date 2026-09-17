import pandas as pd
from classify_intent import classify_message
import time

# Load the cleaned data
df = pd.read_csv('data/clean_pairs.csv')

# Run classification on a larger batch first, so we can sample evenly
# across intents afterward instead of pure random sampling.
# 400 rows is enough to get a spread across all 9 categories.
batch = df.sample(n=400, random_state=7)

predicted_intents = []
for i, row in enumerate(batch.itertuples()):
    intent = classify_message(row.customer_text_clean)
    predicted_intents.append(intent)
    if i % 20 == 0:
        print(f"Classified {i}/{len(batch)}")
    time.sleep(1.2)  # stay under Groq free-tier rate limit

batch['predicted_intent'] = predicted_intents

# Now pull roughly 22-25 examples per intent for the golden set,
# so every category has real representation
golden_candidates = (
    batch.groupby('predicted_intent', group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 22), random_state=7))
)

golden_candidates = golden_candidates[[
    'tweet_id_customer', 'customer_text_clean', 'brand_text_clean', 'predicted_intent'
]]

golden_candidates.to_csv('data/golden_set_candidates.csv', index=False)
print(f"\nSaved {len(golden_candidates)} candidate rows to data/golden_set_candidates.csv")
print("Open this file and manually fill in: true_intent, true_escalate, escalate_reason, good_reply_notes")
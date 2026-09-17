import pandas as pd
from scipy.stats import pearsonr

df = pd.read_csv('data/human_scoring_sample.csv')

criteria = ["relevance", "accuracy", "tone", "completeness"]

print("--- JUDGE vs HUMAN AGREEMENT ---\n")

for criterion in criteria:
    judge_col = criterion
    human_col = f"human_{criterion}"

    # Drop any rows where either score is missing, so they don't break the calculation
    valid_rows = df[[judge_col, human_col]].dropna()

    correlation, p_value = pearsonr(valid_rows[judge_col], valid_rows[human_col])

    # Also compute how often judge and human were within 1 point of each other —
    # an easier-to-explain number alongside the correlation
    within_one_point = (abs(valid_rows[judge_col] - valid_rows[human_col]) <= 1).mean()

    print(f"{criterion}:")
    print(f"  Correlation: {correlation:.2f}")
    print(f"  Within 1 point of each other: {within_one_point:.0%}")
    print()

# Overall average across all 4 criteria combined
all_judge_scores = pd.concat([df[c] for c in criteria])
all_human_scores = pd.concat([df[f"human_{c}"] for c in criteria])
overall_corr, _ = pearsonr(all_judge_scores, all_human_scores)
overall_within_one = (abs(all_judge_scores - all_human_scores) <= 1).mean()

print(f"OVERALL:")
print(f"  Correlation: {overall_corr:.2f}")
print(f"  Within 1 point of each other: {overall_within_one:.0%}")
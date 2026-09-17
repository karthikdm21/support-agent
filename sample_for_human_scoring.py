import pandas as pd

judged = pd.read_csv('data/judged_replies.csv')

# Pick 30 random rows to score by hand — enough to compute a meaningful
# agreement number, small enough to actually do manually
sample = judged.sample(n=30, random_state=42)

# Save with empty columns for you to fill in — mirrors the judge's
# own criteria so scores are directly comparable
sample['human_relevance'] = ''
sample['human_accuracy'] = ''
sample['human_tone'] = ''
sample['human_completeness'] = ''

sample.to_csv('data/human_scoring_sample.csv', index=False)
print(f"Saved {len(sample)} rows to data/human_scoring_sample.csv")
print("Open this file and fill in the human_* columns (1-5 each) WITHOUT looking at the judge's scores first")
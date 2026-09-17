import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Load the cleaned data from step 1
df = pd.read_csv('data/clean_pairs.csv')

# Short messages are usually replies/confirmations mid-conversation
# ("Thank you!", "iOS 11.0.3"), not the actual complaint. They cluster
# together based on length, not meaning, and pollute the real intent groups.
df = df[df['customer_text_clean'].str.len() > 40]

# Clustering all 20k rows is slow and unnecessary just to find categories.
# A random sample is enough to see what kinds of issues customers have.
sample = df.sample(n=3000, random_state=42)

# Turn each customer message into a numeric vector that captures its meaning.
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sample['customer_text_clean'].tolist(), show_progress_bar=True)

# Group the messages into clusters based on our earlier read: hardware,
# software bug, billing, sync/backup, login, other — starting at 6.
NUMBER_OF_CLUSTERS = 12
kmeans = KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=42, n_init=10)
sample['cluster'] = kmeans.fit_predict(embeddings)

sample.to_csv('data/clustered_sample.csv', index=False)
print(f"Saved {len(sample)} rows with cluster labels to data/clustered_sample.csv")

for cluster_id in range(NUMBER_OF_CLUSTERS):
    print(f"\n--- Cluster {cluster_id} ---")
    examples = sample[sample['cluster'] == cluster_id]['customer_text_clean'].head(5)
    for text in examples:
        print(f"  - {text}")
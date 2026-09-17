import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

df = pd.read_csv('data/clean_pairs.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_or_create_collection(name="applesupport_pairs")

print(f"Embedding {len(df)} customer messages...")

embeddings = model.encode(
    df['customer_text_clean'].tolist(),
    show_progress_bar=True
)

# Chroma rejects adding more than ~5461 items in one call, so we add
# in chunks instead of all 71,062 rows at once.
CHUNK_SIZE = 5000
total_rows = len(df)

for start in range(0, total_rows, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, total_rows)

    chunk_ids = [str(i) for i in df.index[start:end]]
    chunk_embeddings = embeddings[start:end].tolist()
    chunk_documents = df['customer_text_clean'].tolist()[start:end]
    chunk_metadatas = [
        {"brand_reply": reply}
        for reply in df['brand_text_clean'].tolist()[start:end]
    ]

    collection.add(
        ids=chunk_ids,
        embeddings=chunk_embeddings,
        documents=chunk_documents,
        metadatas=chunk_metadatas
    )

    print(f"Added rows {start} to {end}")

print(f"Saved {collection.count()} entries to data/chroma_db")
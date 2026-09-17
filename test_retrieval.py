import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection(name="applesupport_pairs")

def retrieve_similar(customer_message, top_k=3):
    query_embedding = model.encode([customer_message]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results

# Test with a real example
test_message = "my phone battery drains so fast since the update"
results = retrieve_similar(test_message)

print(f"Query: {test_message}\n")
for i in range(len(results['documents'][0])):
    similar_complaint = results['documents'][0][i]
    matched_reply = results['metadatas'][0][i]['brand_reply']
    print(f"--- Match {i+1} ---")
    print(f"Similar past complaint: {similar_complaint}")
    print(f"Brand's actual reply: {matched_reply}\n")
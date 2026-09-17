import os
from groq import Groq
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_collection(name="applesupport_pairs")


def retrieve_similar(customer_message, top_k=3):
    query_embedding = embed_model.encode([customer_message]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    similar_pairs = []
    for i in range(len(results['documents'][0])):
        similar_pairs.append({
            "complaint": results['documents'][0][i],
            "reply": results['metadatas'][0][i]['brand_reply']
        })
    return similar_pairs


def generate_reply(customer_message):
    similar_pairs = retrieve_similar(customer_message)

    # Build a text block showing the LLM real past examples of how
    # AppleSupport actually resolved similar issues
    examples_text = ""
    for pair in similar_pairs:
        examples_text += f'\nPast complaint: "{pair["complaint"]}"\n'
        examples_text += f'AppleSupport\'s actual reply: "{pair["reply"]}"\n'

    prompt = f"""You are AppleSupport's customer service agent replying on Twitter/X.

Here are real examples of how AppleSupport has replied to similar past issues:
{examples_text}

Write a reply to this new customer message, in AppleSupport's typical tone —
helpful, concise, asks a clarifying question if needed, never overly formal
or robotic. Keep it under 280 characters, like a real tweet reply.

New customer message: "{customer_message}"

Respond with ONLY the reply text, nothing else."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,  # a little creativity is fine for reply generation, unlike classification
        max_tokens=300,
        reasoning_effort="low"
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    test_message = "my phone battery drains so fast since the update"
    reply = generate_reply(test_message)
    print(f"Customer: {test_message}\n")
    print(f"Generated reply: {reply}")
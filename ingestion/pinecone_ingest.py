import os
import json
import uuid
import time
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cheese_raw.json")
DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cheese_docs_pinecone.jsonl")

client = OpenAI(api_key=OPENAI_API_KEY)

### 🔍 GPT-driven enrichment
def enrich_text_with_gpt(product):
    prompt = f"""
Write a 2-3 sentence product summary suitable for a cheese recommendation chatbot.

Include brand, product type, and a helpful suggestion for how this cheese might be used. Be helpful, friendly, and accurate.

Product info:
- Name: {product.get('name', 'N/A')}
- Price: {product.get('prices', {}).get('Each', 'N/A')} ({product.get('pricePer', '')})
- Brand: {product.get('brand', '')}
- Department: {product.get('department', '')}
    """.strip()

    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GPT enrichment failed: {e}")
        return None

### 📝 Build and save enriched docs.jsonl
def generate_docs_jsonl():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_products = json.load(f)

    with open(DOCS_PATH, "w", encoding="utf-8") as out:
        for product in tqdm(raw_products, desc="Enriching with GPT"):
            text = enrich_text_with_gpt(product)
            if text:
                out.write(json.dumps({
                    "title": product.get("name", ""),
                    "text": text,
                    "case_count": product.get('itemCounts', {}).get('CASE', ''),
                    "each_count": product.get('itemCounts', {}).get('EACH', ''),
                    "case_weight": product.get('weights', {}).get('CASE', ''),
                    "each_weight": product.get('weights', {}).get('EACH', ''),
                    "case_dimensions": product.get('dimensions', {}).get('CASE', ''),
                    "each_dimensions": product.get('dimensions', {}).get('EACH', ''),
                    "each_price": product.get('prices', {}).get('Each', ''),
                    "case_price": product.get('prices', {}).get('Case', ''),
                    "price_per_lb": product.get('pricePer', ''),
                    "brand": product.get("brand", ""),
                    "category": product.get("department", ""),
                    "product_url": product.get("href", ""),
                    "image_url": product.get("showImage", ""),
                    "sku": product.get("sku", ""),
                    "priceOrder": product.get("priceOrder", ""),
                    "popularityOrder": product.get("porpularityOrder", ""),
                    "relateds": product.get("relateds", "")
                }) + "\n")

### 📦 Load enriched docs
def load_docs():
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if "text" in line]

### 🔢 Embedding function
def embed_text(text):
    resp = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return resp.data[0].embedding

### 🚚 Batched upsert
def batched_upsert(index, vectors, namespace="cheese", batch_size=20, retries=3, delay=1.0):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        for attempt in range(1, retries + 1):
            try:
                index.upsert(vectors=batch, namespace=namespace)
                print(f"✅ Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")
                time.sleep(delay)
                break
            except Exception as e:
                print(f"⚠️ Attempt {attempt} failed for batch {i // batch_size + 1}: {e}")
                time.sleep(delay * attempt)
        else:
            print(f"❌ Gave up on batch {i // batch_size + 1} after {retries} retries.")

### 🚀 Main routine
def main():
    generate_docs_jsonl()
    docs = load_docs()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if "cheese-products" not in pc.list_indexes().names():
        pc.create_index(
            name="cheese-products",
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index("cheese-products")

    vectors = []
    for doc in tqdm(docs, desc="Embedding enriched docs"):
        try:
            emb = embed_text(doc["text"])
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": emb,
                "metadata": doc
            })
        except Exception as e:
            print(f"❌ Failed to embed {doc['title']}: {e}")

    if vectors:
        batched_upsert(index=index, vectors=vectors, namespace="cheese")
        print(f"✅ {len(vectors)} enriched vectors upserted to Pinecone.")
    else:
        print("⚠️ No vectors to upsert.")

if __name__ == "__main__":
    main()

import openai
from pinecone import Pinecone

class PineconeCheeseSearch:
    def __init__(self, api_key, env, index_name, openai_api_key):
        # Initialize Pinecone client (new SDK)
        self.client = Pinecone(api_key=api_key, environment=env)
        self.index = self.client.Index(index_name)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)

    def embed_query(self, query):
        # Use OpenAI to embed the query
        response = self.openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def vector_search(self, query, top_k=5):
        vector = self.embed_query(query)
        results = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        return results['matches']

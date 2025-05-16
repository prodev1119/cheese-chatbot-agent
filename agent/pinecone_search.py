import openai
from pinecone import Pinecone

class PineconeCheeseSearch:
    def __init__(self, api_key, env, index_name, openai_api_key):
        # Initialize Pinecone client
        self.client = Pinecone(api_key=api_key, environment=env)
        self.index = self.client.Index(index_name)
        # Correctly initialize OpenAI client with the provided API key
        self.openai_client = openai.OpenAI(api_key=openai_api_key)

    def embed_query(self, query):
        # Use OpenAI to embed the query
        response = self.openai_client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def vector_search(self, original_query):
        # Use LLM to refine the query for better semantic search
        try:
            llm_enhanced_query_prompt = (
                f"A user is searching for cheese products based on their query: '{original_query}'. "
                f"Generate a concise search phrase that best represents the core request for a semantic vector search against a database of cheese product descriptions. "
                f"Focus on key cheese characteristics (type, name, main attributes if mentioned) that would likely be present in the product descriptions. "
                f"- If the query is for a specific cheese type (e.g., 'Cheddar', 'Goat Cheese', 'Brie'), the search phrase should be very direct, like 'Cheddar cheese', 'goat cheese', or 'Brie cheese'. Avoid adding many extra descriptive words unless they are essential differentiators mentioned in the query. The goal is to match documents that explicitly state they are that type of cheese. For example, for '{original_query}', if it's about a specific type, keep the query focused on that type. "
                f"- If the query is broader or describes a use case (e.g., 'cheese for a party', 'cheese for athletes'), then generate a descriptive phrase capturing key attributes (e.g., 'Assortment of popular and versatile cheeses for a party platter', 'Low-fat, high-protein cheese options suitable for athletes'). "
                f"Output only the single, refined search phrase. Do not add conversational filler or explanations."
                )

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at refining search queries for a cheese database."},
                    {"role": "user", "content": llm_enhanced_query_prompt}
                ],
                max_tokens=100,
                temperature=0.5 # Slightly creative but still focused
            )
            enhanced_query = response.choices[0].message.content.strip()
            print(f"Original Pinecone query: '{original_query}', LLM Enhanced query: '{enhanced_query}'")
        except Exception as e:
            print(f"Error enhancing query with LLM: {e}. Falling back to original query.")
            enhanced_query = original_query

        vector = self.embed_query(enhanced_query)
        query_response = self.index.query(vector=vector, top_k=5, include_metadata=True)

        # Extract metadata from matches
        results_with_metadata = []
        if query_response and query_response.get('matches'):
            for match in query_response['matches']:
                if match.get('metadata'):
                    # Optionally, you can include the score if needed by the app
                    # metadata = match['metadata']
                    # metadata['score'] = match['score']
                    # results_with_metadata.append(metadata)
                    results_with_metadata.append(match['metadata'])
                # else: metadata is missing for this match, skip or add placeholder
        return results_with_metadata

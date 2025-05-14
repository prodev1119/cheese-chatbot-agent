class HybridCheeseSearch:
    def __init__(self, mongo_search, pinecone_search):
        self.mongo_search = mongo_search
        self.pinecone_search = pinecone_search

    def search(self, query, top_k=5):
        mongo_results = self.mongo_search.text_search(query, limit=top_k)
        pinecone_results = self.pinecone_search.vector_search(query, top_k=top_k)
        # Combine and deduplicate by product ID or title
        combined = {item['title']: item for item in mongo_results}
        for match in pinecone_results:
            meta = match['metadata']
            if meta['title'] not in combined:
                combined[meta['title']] = meta
        return list(combined.values())[:top_k]

from pymongo import MongoClient

class MongoCheeseSearch:
    def __init__(self, uri, db_name, collection_name):
        self.client = MongoClient(uri)
        self.collection = self.client[db_name][collection_name]

    def text_search(self, query, limit=5):
        # Simple text search (can use $text if indexed)
        results = self.collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        return list(results)

client = MongoClient("mongodb://localhost:27017/")
db = client["cheese_db"]
db.cheeses.create_index([("title", "text"), ("text", "text")])

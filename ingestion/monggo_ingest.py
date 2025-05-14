import json
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['cheese_db']
collection = db['cheeses']

with open('data/cheese_docs_monggo.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        doc = json.loads(line)
        collection.insert_one(doc)

print("Data inserted into MongoDB!")

from pymongo import MongoClient
from pymongo.errors import OperationFailure
import os

class MongoCheeseSearch:
    def __init__(self, uri, db_name, collection_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def search(self, query_input: dict | list):
        print(f"Searching for: {query_input}")
        results = []
        # Determine if it's an aggregation query or a find query
        is_aggregation = False
        if isinstance(query_input, list): # Standard way to define an aggregation pipeline
            is_aggregation = True
        elif isinstance(query_input, dict):
            # Check for common aggregation stage operators at the top level of the dict
            # This is a heuristic if the LLM outputs a single stage as a dict instead of a list
            # or if it doesn't use the $query, $orderby, $limit structure for a find query.
            aggregation_operators = {'$group', '$match', '$sort', '$limit', '$project', '$unwind', '$count', '$addFields', '$lookup'}
            if any(op in query_input for op in aggregation_operators) and not ('$query' in query_input or '$orderby' in query_input or '$text' in query_input):
                # If it has aggregation operators and isn't a find query with $text or our custom structure
                # Treat as a single-stage aggregation pipeline
                is_aggregation = True
                query_input = [query_input] # Wrap in a list for the aggregate method

        if is_aggregation:
            try:
                cursor = self.collection.aggregate(query_input)
                results = list(cursor)
            except Exception as e:
                print(f"Error during MongoDB aggregation: {e}. Query: {query_input}")
                return [] # Return empty list on aggregation error
        else: # It's a find query
            actual_query = query_input
            sort_params = None
            limit_val = None

            if isinstance(query_input, dict) and ('$query' in query_input or '$orderby' in query_input or '$limit' in query_input):
                actual_query = query_input.get("$query", {})
                sort_params = query_input.get("$orderby")
                limit_val = query_input.get("$limit")
            elif not actual_query and not ('$query' in query_input or '$orderby' in query_input or '$limit' in query_input):
                actual_query = query_input

            try:
                cursor = self.collection.find(actual_query)
                if sort_params:
                    sort_list = []
                    if isinstance(sort_params, dict):
                        for field, order in sort_params.items():
                            try:
                                order_val = int(order)
                                if order_val == 1 or order_val == -1:
                                    sort_list.append((field, order_val))
                                else:
                                    print(f"Warning: Invalid sort order '{order}' for field '{field}'. Ignoring.")
                            except ValueError:
                                print(f"Warning: Sort order for field '{field}' not an int: {order}. Ignoring.")
                        if sort_list:
                            cursor = cursor.sort(sort_list)
                    else:
                        print(f"Warning: '$orderby' not a dict: {sort_params}. Ignoring sort.")

                if limit_val is not None:
                    try:
                        cursor = cursor.limit(int(limit_val))
                    except ValueError:
                        print(f"Warning: Could not convert '$limit' to int: {limit_val}. Ignoring limit.")
                results = list(cursor)
            except Exception as e:
                print(f"Error during MongoDB find operation: {e}. Query: {actual_query}")
                return [] # Return empty list on find error

        # Convert ObjectId to str for JSON serialization
        for result in results:
            if "_id" in result and hasattr(result["_id"], "__str__") and not isinstance(result["_id"], str):
                result["_id"] = str(result["_id"])
        return results

# One-time setup or ensure index exists
try:
    setup_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    setup_db = setup_client[os.getenv("DB_NAME", "cheese_db")]
    setup_collection = setup_db[os.getenv("COLLECTION_NAME", "cheeses")]

    setup_collection.create_index(
        [("title", "text"), ("text", "text")],
        name="title_text_index",
        default_language="english"
    )
    print("Text index 'title_text_index' ensured on 'cheeses' collection.")
except OperationFailure as e:
    if e.code == 85:
        print(f"Could not create index 'title_text_index' due to conflict: {e.details['errmsg']}. Assuming functional index already exists.")
    elif e.code == 86:
        print(f"Could not create index 'title_text_index' due to key specs conflict: {e.details['errmsg']}. An index with the same name but different keys might exist.")
    else:
        print(f"A MongoDB operation error occurred during index creation (Code {e.code}): {e.details['errmsg']}")
except Exception as e:
    print(f"An unexpected error occurred during MongoDB index setup: {e}")
finally:
    if 'setup_client' in locals():
        setup_client.close()

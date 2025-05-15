import json
import re
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Any, Literal
from openai import OpenAI

# Updated state definition
class CheeseAgentState(TypedDict):
    input: str
    results: List[Any]
    is_cheese_related: bool | None
    mongo_query: str | None # The JSON string from the LLM
    is_aggregation_result: bool | None # Flag from mongo_search_node
    final_response: str | None
    history: List[str]

def build_cheese_agent(mongo_search, pinecone_search, openai_api_key):
    client = OpenAI(api_key=openai_api_key)

    # Node functions
    def check_cheese_related_node(state: CheeseAgentState) -> dict:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that determines if a user's query is about cheese."},
                {"role": "user", "content": f"Is the following query about cheese? Only respond with 'yes' or 'no': '{state['input']}'"}
            ],
            max_tokens=10
        )
        is_cheese = "yes" in response.choices[0].message.content.strip().lower()
        new_history = state.get("history", []) + ["check_cheese_related_node"]
        return {
            "is_cheese_related": is_cheese,
            "history": new_history,
            "results": [],
            "mongo_query": None,
            "is_aggregation_result": None # Reset flag
            }

    def convert_to_mongo_query_node(state: CheeseAgentState) -> dict:
        if not state.get("is_cheese_related"):
            new_history = state.get("history", []) + ["convert_to_mongo_query_node_skipped"]
            return {"history": new_history, "mongo_query": None, "is_aggregation_result": None}

        user_query_for_llm = state['input']

        system_prompt_template = (
            "You are a helpful assistant that converts a user's query to a MongoDB query. MongoDB data is about Cheese products. "
            "CRITICAL: Output ONLY the raw MongoDB query as a valid JSON string. NO MARKDOWN, NO SINGLE QUOTES WRAPPING THE JSON, NO EXPLANATIONS. "
            "The output must be directly parsable by a JSON loader. "
            "1. For simple filtering, text searches, or when specific fields are mentioned for retrieval (e.g., 'show me all cheddar cheese', 'cheddar cheese under $5'), use a find query structure (a JSON object). "
            "   - For text searches: {example_text_search}. "
            "   - For exact field matches: {example_field_match}. "
            "   - For comparisons (e.g., price less than 50): {example_comparison_query}. "
            "2. If the query implies sorting or limiting specific documents (e.g., 'most popular cheese', 'cheapest 5 cheeses'), use the find query structure with '$query', '$orderby', and '$limit' keys (a JSON object). "
            "   - Example for 'most popular cheese': {example_most_popular}. "
            "   - {price_logic_placeholder} "
            "3. If the query requires counting distinct items, grouping, or other complex aggregations (e.g., 'how many brands', 'count of categories'), provide an aggregation pipeline as a JSON array of stages. "
            "   - Example for 'how many brands of cheese do you have?': {example_count_distinct_brands}. "
            "   - Example for 'how many cheese products in total?': {example_total_count}. "
            "4. If the query is for a specific cheese type (e.g., 'Cheddar', 'Goat Cheese', 'Brie'), create a query to recommend cheese products with content same as the type of cheese of the user's query"
            "   - Example for 'can you show me all goat cheese?': {example_goat_search}. "
            "MongoDB Data fields: title, text, each_price (price of one item), brand, category, "
            "priceOrder (integer, higher value is CHEAPER, lower value is MORE EXPENSIVE), "
            "popularityOrder (integer, higher value is MORE POPULAR)."
        )

        examples = {
            "example_text_search": '{\"$text\": {\"$search\": \"blue cheese\"}}',
            "example_field_match": '{\"$text\": {\"$search\": \"cheddar\"}}',
            "example_comparison_query": '{\"each_price\": {\"$lt\": 50}}',
            "example_most_popular": '{\"$query\": {}, \"$orderby\": {\"popularityOrder\": -1}, \"$limit\": 1}',
            "cheapest_cheese_query": '{\"$query\": {}, \"$orderby\": {\"priceOrder\": -1}, \"$limit\": 1}',
            "most_expensive_10_query": '{\"$query\": {}, \"$orderby\": {\"priceOrder\": 1}, \"$limit\": 10}',
            "example_count_distinct_brands": '[{\"$group\": {\"_id\": \"$brand\"}}, {\"$count\": \"unique_brands\"}]',
            "example_total_count": '[{\"$count\": \"total_products\"}]',
            "example_goat_search": '{\"$text\": {\"$search\": \"goat\"}}'
        }

        price_logic_str = ""
        if "cheapest" in user_query_for_llm.lower() or "lowest price" in user_query_for_llm.lower():
            price_logic_str = f"   - For 'cheapest cheese': {examples['cheapest_cheese_query']}. "
        elif "most expensive" in user_query_for_llm.lower() or "highest price" in user_query_for_llm.lower():
            if "10" in user_query_for_llm or "ten" in user_query_for_llm.lower():
                price_logic_str = f"   - For 'most expensive 10 cheeses': {examples['most_expensive_10_query']}. "
            else:
                _json_example = '{\"$query\":{}, \"$orderby\":{\"priceOrder\":1}, \"$limit\":1}'
                price_logic_str = f"   - For 'most expensive cheese': {_json_example}. "

        system_prompt = system_prompt_template.format(
            example_text_search=examples['example_text_search'],
            example_field_match=examples['example_field_match'],
            example_comparison_query=examples['example_comparison_query'],
            example_most_popular=examples['example_most_popular'],
            price_logic_placeholder=price_logic_str,
            example_count_distinct_brands=examples['example_count_distinct_brands'],
            example_total_count=examples['example_total_count'],
            example_goat_search=examples['example_goat_search']
        ).replace("  ", " ")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Convert this cheese-related query to a MongoDB query: '{user_query_for_llm}'"}
            ],
            max_tokens=300
        )
        mongo_query_json_str = response.choices[0].message.content.strip()
        new_history = state.get("history", []) + ["convert_to_mongo_query_node"]
        return {"mongo_query": mongo_query_json_str, "history": new_history, "is_aggregation_result": None}

    def mongo_search_node(state: CheeseAgentState) -> dict:
        mongo_query_json_str = state.get("mongo_query")
        current_search_results = []
        is_aggregation = False # Default

        if not mongo_query_json_str:
            new_history = state.get("history", []) + ["mongo_search_node_skipped_no_query"]
            return {"results": [], "history": new_history, "is_aggregation_result": False}

        cleaned_json_str = mongo_query_json_str
        match = re.match(r"^\s*```(?:json)?\s*([\s\S]+?)\s*```\s*$", cleaned_json_str, re.DOTALL)
        if match:
            cleaned_json_str = match.group(1).strip()

        if cleaned_json_str.startswith("'") and cleaned_json_str.endswith("'"):
            cleaned_json_str = cleaned_json_str[1:-1]
        elif cleaned_json_str.startswith('"') and cleaned_json_str.endswith('"'):
            # Be careful if the JSON string itself is supposed to be a string like ""hello""
            if not (len(cleaned_json_str) > 1 and cleaned_json_str[1] == '\"' and cleaned_json_str[-2] == '\"') :
                cleaned_json_str = cleaned_json_str[1:-1]


        if not cleaned_json_str:
            print(f"mongo_query_json_str was present but became empty after stripping: '{mongo_query_json_str}'")
            new_history = state.get("history", []) + ["mongo_search_node_error_empty_after_strip"]
            search_output = mongo_search.search({"$text": {"$search": state['input']}}) # Fallback
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            return {"results": current_search_results, "history": new_history, "is_aggregation_result": False}

        try:
            query_input = json.loads(cleaned_json_str)
            # Determine if it was intended as aggregation for flag setting
            if isinstance(query_input, list): # Aggregation pipelines are lists of stages
                is_aggregation = True
            elif isinstance(query_input, dict) and any(op.startswith('$') for op in query_input.keys()) and not any(k in query_input for k in ['$query', '$orderby', '$limit', '$text']):
                # Heuristic: if dict has operators like $group, $count, and not our find structure, assume aggregation
                is_aggregation = True

            search_output = mongo_search.search(query_input) # mongo_search.search handles list or dict
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []

            # Refine is_aggregation based on typical aggregation result structure (e.g., a list with one dict having a count)
            if not is_aggregation and len(current_search_results) == 1 and isinstance(current_search_results[0], dict):
                # Common aggregation results are like [{'count_field': 10}]
                # Product results usually have 'title', 'brand', etc.
                result_keys = current_search_results[0].keys()
                if not any(key in result_keys for key in ['title', 'brand', 'each_price', 'text']): # If it doesn't look like a product
                    if any(key.endswith('count') or key.endswith('Count') or key.startswith('total_') or key.startswith('unique_') for key in result_keys):
                        is_aggregation = True


        except json.JSONDecodeError as e:
            print(f"Error decoding mongo_query_json_str: Original='{mongo_query_json_str}', Cleaned='{cleaned_json_str}'. Error: {e}")
            search_output = mongo_search.search({"$text": {"$search": state['input']}}) # Fallback
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            is_aggregation = False # Fallback is a find query
        except Exception as e:
            print(f"An unexpected error occurred in mongo_search_node: {e}. Original query string: '{mongo_query_json_str}'")
            search_output = mongo_search.search({"$text": {"$search": state['input']}}) # Fallback
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            is_aggregation = False # Fallback is a find query

        new_history = state.get("history", []) + ["mongo_search_node"]
        return {"results": current_search_results, "history": new_history, "is_aggregation_result": is_aggregation}

    def pinecone_search_node(state: CheeseAgentState) -> dict: # Pinecone always returns products, not aggregation
        query = state["input"]
        search_output = pinecone_search.vector_search(query)
        current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
        new_history = state.get("history", []) + ["pinecone_search_node"]
        return {"results": current_search_results, "history": new_history, "is_aggregation_result": False}

    def generate_response_node(state: CheeseAgentState) -> dict:
        input_query = state["input"]
        results = state.get("results", [])
        is_cheese = state.get("is_cheese_related")
        mongo_query_generated_str = state.get("mongo_query")
        is_aggregation_res = state.get("is_aggregation_result", False) # Get the flag
        history = state.get("history", [])

        prompt_parts = [f"User query: '{input_query}'."]

        if is_cheese is False:
            prompt_parts.append("This query is not cheese-related. Provide a general response or a polite refusal if you cannot assist.")
        elif is_cheese is True:
            prompt_parts.append("This query is cheese-related.")
            last_search_node = None
            relevant_history = [h for h in history if h not in ["generate_response_node", "check_cheese_related_node", "convert_to_mongo_query_node"]]
            if relevant_history:
                node_that_produced_results = relevant_history[-1]
                if "mongo_search_node" in node_that_produced_results: last_search_node = "mongo_search_node"
                elif "pinecone_search_node" in node_that_produced_results: last_search_node = "pinecone_search_node"

            if is_aggregation_res and results and isinstance(results[0], dict):
                # Handle aggregation result (typically a list with one dict, e.g. [{'count': 10}])
                prompt_parts.append(f"The MongoDB query (derived from LLM: '{mongo_query_generated_str}') was an aggregation. The result is:")
                prompt_parts.append(json.dumps(results[0], indent=2))
                prompt_parts.append("Based on this aggregation result, provide a direct answer to the user's query. Do NOT list product examples unless the aggregation itself returns product-like fields (which is rare for counts/totals).")
            elif results: # It's a list of products
                if last_search_node == "mongo_search_node":
                    prompt_parts.append(f"Found {len(results)} product(s) in MongoDB (derived query from LLM: '{mongo_query_generated_str}').")
                elif last_search_node == "pinecone_search_node":
                    prompt_parts.append(f"MongoDB found no results (derived query: '{mongo_query_generated_str}'). Pinecone Vector Search then found {len(results)} product(s) based on the original query.")
                else: prompt_parts.append(f"Found {len(results)} product(s).")
                prompt_parts.append("Based on these products, generate a helpful response. List a few examples if appropriate.")
                MAX_RESULTS_IN_PROMPT = 5 # Reduced for brevity with aggregation handling
                results_for_prompt = results[:MAX_RESULTS_IN_PROMPT]
                try: prompt_parts.append(json.dumps(results_for_prompt, indent=2))
                except TypeError as te:
                    prompt_parts.append(f"(Could not serialize some results to JSON for prompt: {te})")
                    prompt_parts.append(str(results_for_prompt))
                if len(results) > MAX_RESULTS_IN_PROMPT:
                    prompt_parts.append(f"...and {len(results) - MAX_RESULTS_IN_PROMPT} more products not shown in this context.")
            else: # No results
                if "mongo_search_node_skipped_no_query" in history:
                    prompt_parts.append("The step to query MongoDB was skipped because no MongoDB query was generated. This usually means the query was not cheese-related initially.")
                    prompt_parts.append("Politely inform the user you can only help with cheese related questions or provide a general greeting if unsure.")
                elif last_search_node == "pinecone_search_node":
                    prompt_parts.append(f"After searching MongoDB (derived query: '{mongo_query_generated_str}') and then Pinecone (semantic search based on the original user query '{input_query}'), no specific cheese products matching your request were found.")
                    prompt_parts.append(f"First, acknowledge that you couldn't find specific products for '{input_query}'. "
                                        f"Then, using your general knowledge of cheese, suggest 1-2 alternative *types* of cheese that someone looking for '{input_query}' might also enjoy, briefly explaining why (e.g., similar taste profile, texture, origin). For example, if the user asked for 'goat cheese' and none was found, you might suggest Feta or a Sheep's milk cheese as alternatives with similar tanginess or texture. "
                                        f"Finally, offer to help in other ways, such as by asking for more details about what kind of cheese they are looking for (e.g., taste, texture, occasion), or suggesting some general cheese categories they might be interested in exploring (e.g., soft cheeses, hard cheeses, blue cheeses). Make the response conversational and helpful.")
                elif last_search_node == "mongo_search_node":
                    prompt_parts.append(f"Searched MongoDB (derived query from LLM: '{mongo_query_generated_str}') but found no relevant cheese products. Pinecone search was not attempted as per workflow (this path means Mongo should have results or Pinecone is next). This specific scenario of no mongo results and no pinecone search should be rare if routing is correct, but handle gracefully.")
                    prompt_parts.append(f"Inform the user that no products for '{input_query}' were found. You can ask for clarification or suggest they rephrase.")
                else: # Should ideally not be reached if last_search_node is always set when a search happens
                    prompt_parts.append("No search results are available for this cheese query after attempting searches.")
                    prompt_parts.append(f"Politely inform the user that you couldn't find information for '{input_query}'. You can ask if they want to try a different query.")
        else:
            prompt_parts.append("Could not determine if the query is cheese-related or an issue occurred before search.")

        llm_final_response_messages = [
                {"role": "system", "content": "You are a helpful cheese assistant. Synthesize the information provided into a concise, helpful, and easy-to-read answer. "
                                            "Ensure proper formatting and spacing. If results are a direct answer to a question (like a count), state it clearly. "
                                            "If results are products, list the examples if appropriate."
                                            "Do not run words together."
                                            },
                {"role": "user", "content": "\n".join(prompt_parts)}
            ]

        llm_final_response = client.chat.completions.create(
            model="gpt-4o", messages=llm_final_response_messages, max_tokens=500
        ).choices[0].message.content.strip()

        current_history = state.get("history", [])
        new_history = current_history + ["generate_response_node"]

        # Determine what results to pass to Streamlit
        final_results_for_streamlit = results
        if is_aggregation_res:
            final_results_for_streamlit = [] # Clear results if aggregation

        return {
            "final_response": llm_final_response,
            "history": new_history,
            "results": final_results_for_streamlit, # Use modified list
            "is_aggregation_result": is_aggregation_res
        }

    # Routing functions
    def route_after_check_cheese(state: CheeseAgentState) -> Literal["convert_to_mongo_query", "generate_response"]:
        if state.get("is_cheese_related"):
            return "convert_to_mongo_query"
        else:
            return "generate_response"

    def route_after_mongo_search(state: CheeseAgentState) -> Literal["pinecone_search", "generate_response"]:
        if state.get("results") and len(state.get("results", [])) > 0:
            return "generate_response"
        else:
            return "pinecone_search"

    graph = StateGraph(CheeseAgentState)
    graph.add_node("check_cheese_related", check_cheese_related_node)
    graph.add_node("convert_to_mongo_query", convert_to_mongo_query_node)
    graph.add_node("mongo_search", mongo_search_node)
    graph.add_node("pinecone_search", pinecone_search_node)
    graph.add_node("generate_response", generate_response_node)
    graph.set_entry_point("check_cheese_related")
    graph.add_conditional_edges(
        "check_cheese_related",
        route_after_check_cheese,
        {
            "convert_to_mongo_query": "convert_to_mongo_query",
            "generate_response": "generate_response"
        }
    )
    graph.add_edge("convert_to_mongo_query", "mongo_search")
    graph.add_conditional_edges(
        "mongo_search",
        route_after_mongo_search,
        {
            "generate_response": "generate_response",
            "pinecone_search": "pinecone_search"
        }
    )
    graph.add_edge("pinecone_search", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()
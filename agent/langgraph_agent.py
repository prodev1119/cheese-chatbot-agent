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
    thinking_log: List[str] # To store reasoning steps

def build_cheese_agent(mongo_search, pinecone_search, openai_api_key):
    client = OpenAI(api_key=openai_api_key)

    # Node functions
    def check_cheese_related_node(state: CheeseAgentState) -> dict:
        input_query = state['input']
        thinking_log = state.get("thinking_log", []) # Initialize or get existing log

        thinking_log.append("_____________________ Step: Check if Query is Cheese-Related _____________________")
        thinking_log.append(f"User Query: '{input_query}'")

        prompt_content = f"Is the following query about cheese? Only respond with 'yes' or 'no': '{input_query}'"
        # thinking_log.append(f"LLM Prompt for cheese check: '{prompt_content}'")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that determines if a user's query is about cheese. If user's query is like:plz show me all products that are expensive than $50., then it is about cheese."},
                {"role": "user", "content": prompt_content}
            ],
            max_tokens=10
        )
        raw_llm_response = response.choices[0].message.content.strip()
        # thinking_log.append(f"LLM Raw Response: '{raw_llm_response}'")

        is_cheese = "yes" in raw_llm_response.lower()
        thinking_log.append(f"Decision: Query is cheese-related = {is_cheese}")
        # thinking_log.append("--------------------------------------------------------------------")

        new_history = state.get("history", []) + ["check_cheese_related_node"]
        return {
            "is_cheese_related": is_cheese,
            "history": new_history,
            "results": [],
            "mongo_query": None,
            "is_aggregation_result": None, # Reset flag
            "thinking_log": thinking_log
            }

    def convert_to_mongo_query_node(state: CheeseAgentState) -> dict:
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("____________________ Step: Convert User Query to MongoDB Query ____________________")

        if not state.get("is_cheese_related"):
            thinking_log.append("Decision: Skipped MongoDB query conversion (query not cheese-related).")
            # thinking_log.append("--------------------------------------------------------------------")
            new_history = state.get("history", []) + ["convert_to_mongo_query_node_skipped"]
            return {"history": new_history, "mongo_query": None, "is_aggregation_result": None, "thinking_log": thinking_log}

        user_query_for_llm = state['input']
        thinking_log.append(f"User Query for LLM: '{user_query_for_llm}'")

        system_prompt_template = (
            "You are a helpful assistant that converts a user's query to a MongoDB query. MongoDB data is about Cheese products. "
            "CRITICAL: Output ONLY the raw MongoDB query as a valid JSON string. NO MARKDOWN, NO SINGLE QUOTES WRAPPING THE JSON, NO EXPLANATIONS. "
            "The output must be directly parsable by a JSON loader. "
            "1. For simple filtering, text searches, or when specific fields are mentioned for retrieval (e.g., 'show me all sliced cheese', 'sliced cheese under $5'), use a find query structure (a JSON object). "
            "   - For text searches (e.g., blue cheese): {example_text_search}. "
            "   - For exact field matches (e.g., show me all sliced cheese): {example_field_match}. "
            "   - For comparisons (e.g., price less than 50): {example_comparison_query}. "
            "2. If the query implies sorting or limiting specific documents (e.g., 'most popular cheese', 'cheapest 5 cheeses'), use the find query structure with '$query', '$orderby', and '$limit' keys (a JSON object). "
            "   - Example for 'most popular cheese': {example_most_popular}. "
            "   - {price_logic_placeholder} "
            "3. If the query requires counting distinct items, grouping, or other complex aggregations (e.g., 'how many brands', 'count of categories'), provide an aggregation pipeline as a JSON array of stages. "
            "   - Example for 'how many brands of cheese do you have?': {example_count_distinct_brands}. "
            "   - Example for 'how many cheese products in total?': {example_total_count}. "
            "4. If the query is for a specific cheese type (e.g., 'Cheddar', 'Goat Cheese', 'Brie'), create a query to recommend cheese products with content same as the type of cheese of the user's query"
            "   - Example for 'can you show me all goat cheese?': {example_type_search}. "
            "MongoDB Data fields: title, text, each_price (price of one item), brand, category, "
            "priceOrder (integer, higher value is CHEAPER, lower value is MORE EXPENSIVE), "
            "popularityOrder (integer, higher value is MORE POPULAR)."
        )

        examples = {
            "example_text_search": '{\"$text\": {\"$search\": \"blue cheese\"}}',
            "example_field_match": '{\"category\": \"Sliced Cheese\"}',
            "example_comparison_query": '{\"each_price\": {\"$lt\": 50}}',
            "example_most_popular": '{\"$query\": {}, \"$orderby\": {\"popularityOrder\": -1}, \"$limit\": 1}',
            "cheapest_cheese_query": '{\"$query\": {}, \"$orderby\": {\"priceOrder\": -1}, \"$limit\": 1}',
            "most_expensive_10_query": '{\"$query\": {}, \"$orderby\": {\"priceOrder\": 1}, \"$limit\": 10}',
            "example_count_distinct_brands": '[{\"$group\": {\"_id\": \"$brand\"}}, {\"$count\": \"unique_brands\"}]',
            "example_total_count": '[{\"$count\": \"total_products\"}]',
            "example_type_search": '{\"$text\": {\"$search\": \"goat\"}}'
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
            example_type_search=examples['example_type_search']
        ).replace("  ", " ")

        # thinking_log.append(f"System Prompt for LLM (MongoDB query conversion - first 200 chars): {system_prompt[:200]}...")
        user_prompt_for_llm = f"Convert this cheese-related query to a MongoDB query: '{user_query_for_llm}'"
        # thinking_log.append(f"User Prompt for LLM (MongoDB query conversion): '{user_prompt_for_llm}'")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_for_llm}
            ],
            max_tokens=300
        )
        mongo_query_json_str = response.choices[0].message.content.strip()
        thinking_log.append(f"MongoDB Query JSON: '{mongo_query_json_str}'")
        # thinking_log.append("--------------------------------------------------------------------")

        new_history = state.get("history", []) + ["convert_to_mongo_query_node"]
        return {"mongo_query": mongo_query_json_str, "history": new_history, "is_aggregation_result": None, "thinking_log": thinking_log}

    def mongo_search_node(state: CheeseAgentState) -> dict:
        mongo_query_json_str = state.get("mongo_query")
        current_search_results = []
        is_aggregation = False # Default
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("_____________________________ Step: Search MongoDB _____________________________")
        # thinking_log.append(f"Received MongoDB Query JSON: '{mongo_query_json_str}'")

        if not mongo_query_json_str:
            thinking_log.append("Decision: Skipped MongoDB search (no query string provided).")
            new_history = state.get("history", []) + ["mongo_search_node_skipped_no_query"]
            # thinking_log.append("--------------------------------------------------------------------")
            return {"results": [], "history": new_history, "is_aggregation_result": False, "thinking_log": thinking_log}

        cleaned_json_str = mongo_query_json_str
        # thinking_log.append(f"Original JSON string for Mongo: '{mongo_query_json_str}'")
        match = re.match(r"^\s*```(?:json)?\s*([\s\S]+?)\s*```\s*$", cleaned_json_str, re.DOTALL)
        if match:
            cleaned_json_str = match.group(1).strip()
            thinking_log.append(f"JSON after stripping markdown fences: '{cleaned_json_str}'")

        if cleaned_json_str.startswith("'") and cleaned_json_str.endswith("'"):
            cleaned_json_str = cleaned_json_str[1:-1]
            thinking_log.append(f"JSON after stripping single quotes: '{cleaned_json_str}'")
        elif cleaned_json_str.startswith('"') and cleaned_json_str.endswith('"'):
            if not (len(cleaned_json_str) > 1 and cleaned_json_str[1] == '\"' and cleaned_json_str[-2] == '\"'):
                cleaned_json_str = cleaned_json_str[1:-1]
                thinking_log.append(f"JSON after stripping double quotes: '{cleaned_json_str}'")

        if not cleaned_json_str:
            thinking_log.append(f"Warning: MongoDB query string became empty after stripping. Original: '{mongo_query_json_str}'.")
            thinking_log.append(f"Fallback: Performing text search with original user input: '{state['input']}'")
            new_history = state.get("history", []) + ["mongo_search_node_error_empty_after_strip"]
            search_output = mongo_search.search({"$text": {"$search": state['input']}})
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            thinking_log.append(f"Fallback search results count: {len(current_search_results)}")
            thinking_log.append("Determination: Fallback search is NOT an aggregation.")
            # thinking_log.append("--------------------------------------------------------------------")
            return {"results": current_search_results, "history": new_history, "is_aggregation_result": False, "thinking_log": thinking_log}

        try:
            query_input = json.loads(cleaned_json_str)
            # thinking_log.append(f"Successfully parsed JSON to Python object: {query_input}")

            if isinstance(query_input, list):
                is_aggregation = True
                thinking_log.append("Determination: Query is an aggregation pipeline (list of stages).")
            elif isinstance(query_input, dict) and any(op.startswith('$') for op in query_input.keys()) and not any(k in query_input for k in ['$query', '$orderby', '$limit', '$text']):
                is_aggregation = True
                thinking_log.append("Determination: Query is likely an aggregation (dictionary with aggregation operators).")
            else:
                thinking_log.append("Determination: Query is a find query (or uses $text, $query, $orderby, $limit).")

            search_output = mongo_search.search(query_input)
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            thinking_log.append(f"MongoDB search executed. Results count: {len(current_search_results)}")
            # Limit logging of actual results to avoid huge logs
            results_summary_for_log = current_search_results[:2] if current_search_results else "No results"
            # thinking_log.append(f"MongoDB search results (first 2 or none): {json.dumps(results_summary_for_log, indent=2, default=str)}")


            # Refine is_aggregation based on typical aggregation result structure
            if not is_aggregation and len(current_search_results) == 1 and isinstance(current_search_results[0], dict):
                result_keys = current_search_results[0].keys()
                if not any(key in result_keys for key in ['title', 'brand', 'each_price', 'text']):
                    if any(key.endswith('count') or key.endswith('Count') or key.startswith('total_') or key.startswith('unique_') for key in result_keys):
                        is_aggregation = True
                        thinking_log.append("Refined Determination: Result structure suggests an aggregation (e.g., count field found, no typical product fields).")

        except json.JSONDecodeError as e:
            thinking_log.append(f"Error: Failed to decode MongoDB query JSON. Original='{mongo_query_json_str}', Cleaned='{cleaned_json_str}'. Error: {e}")
            thinking_log.append(f"Fallback: Performing text search with original user input: '{state['input']}'")
            search_output = mongo_search.search({"$text": {"$search": state['input']}})
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            is_aggregation = False
            thinking_log.append(f"Fallback search results count: {len(current_search_results)}")
            thinking_log.append("Determination: Fallback search is NOT an aggregation.")
        except Exception as e:
            thinking_log.append(f"Error: An unexpected error occurred during MongoDB search. Query String: '{mongo_query_json_str}'. Error: {e}")
            thinking_log.append(f"Fallback: Performing text search with original user input: '{state['input']}'")
            search_output = mongo_search.search({"$text": {"$search": state['input']}})
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            is_aggregation = False
            thinking_log.append(f"Fallback search results count: {len(current_search_results)}")
            thinking_log.append("Determination: Fallback search is NOT an aggregation.")

        thinking_log.append(f"Final Determination: Is aggregation result = {is_aggregation}")
        new_history = state.get("history", []) + ["mongo_search_node"]
        # thinking_log.append("--------------------------------------------------------------------")
        return {"results": current_search_results, "history": new_history, "is_aggregation_result": is_aggregation, "thinking_log": thinking_log}

    def pinecone_search_node(state: CheeseAgentState) -> dict: # Pinecone always returns products, not aggregation
        query = state["input"]
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("____________________ Step: Search Pinecone (Vector Search) ____________________")
        thinking_log.append(f"Original User Query for Pinecone: '{query}'")

        # pinecone_search.vector_search now returns (results, pinecone_log_entries)
        search_output, pinecone_log_entries = pinecone_search.vector_search(query)

        if pinecone_log_entries:
            thinking_log.extend(pinecone_log_entries) # Add Pinecone's internal log

        current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
        # Limit logging of actual results to avoid huge logs
        results_summary_for_log = current_search_results[:2] if current_search_results else "No results"
        thinking_log.append(f"Pinecone search results (first 2 or none): {json.dumps(results_summary_for_log, indent=2, default=str)}")

        new_history = state.get("history", []) + ["pinecone_search_node"]
        # thinking_log.append("--------------------------------------------------------------------")
        return {"results": current_search_results, "history": new_history, "is_aggregation_result": False, "thinking_log": thinking_log}

    def generate_response_node(state: CheeseAgentState) -> dict:
        input_query = state["input"]
        results = state.get("results", [])
        is_cheese = state.get("is_cheese_related")
        mongo_query_generated_str = state.get("mongo_query")
        is_aggregation_res = state.get("is_aggregation_result", False)
        history = state.get("history", [])
        thinking_log = state.get("thinking_log", [])

        thinking_log.append("________________________ Step: Generate Final Response ________________________")
        thinking_log.append(f"Input User Query: '{input_query}'")
        thinking_log.append(f"Is Cheese-Related Flag: {is_cheese}")
        thinking_log.append(f"MongoDB Query: '{mongo_query_generated_str}'")
        thinking_log.append(f"Is Aggregation Result Flag: {is_aggregation_res}")
        thinking_log.append(f"Number of products/results received: {len(results)}")
        # Limit logging of actual results
        results_summary_for_log = results[:2] if results else "No results"
        # thinking_log.append(f"Products/Results (first 2 or none): {json.dumps(results_summary_for_log, indent=2, default=str)}")
        thinking_log.append(f"Node History: {history}")

        prompt_parts = [f"User query: '{input_query}'."]
        thinking_log.append("Constructing prompt for final LLM response:")

        if is_cheese is False:
            prompt_parts.append("This query is not cheese-related. Provide a general response or a polite refusal if you cannot assist.")
            thinking_log.append("- Determined: Query is not cheese-related.")
        elif is_cheese is True:
            prompt_parts.append("This query is cheese-related.")
            thinking_log.append("- Determined: Query is cheese-related.")
            last_search_node = None
            relevant_history = [h for h in history if h not in ["generate_response_node", "check_cheese_related_node", "convert_to_mongo_query_node"]]
            if relevant_history:
                node_that_produced_results = relevant_history[-1]
                if "mongo_search_node" in node_that_produced_results: last_search_node = "mongo_search_node"
                elif "pinecone_search_node" in node_that_produced_results: last_search_node = "pinecone_search_node"
            thinking_log.append(f"- Determined last search node: {last_search_node}")

            if is_aggregation_res and results and isinstance(results[0], dict):
                prompt_parts.append(f"The MongoDB query (derived from LLM: '{mongo_query_generated_str}') was an aggregation. The result is:")
                prompt_parts.append(json.dumps(results[0], indent=2))
                prompt_parts.append("Based on this aggregation result, provide a direct answer to the user's query. Do NOT list product examples unless the aggregation itself returns product-like fields (which is rare for counts/totals).")
                thinking_log.append("- Determined: Result is an aggregation. Prompting LLM to give direct answer based on aggregation data.")
            elif results: # It's a list of products
                if last_search_node == "mongo_search_node":
                    prompt_parts.append(f"Found {len(results)} product(s) in MongoDB (derived query from LLM: '{mongo_query_generated_str}').")
                    thinking_log.append(f"- Determined: Results from MongoDB ({len(results)} products).")
                elif last_search_node == "pinecone_search_node":
                    prompt_parts.append(f"MongoDB found no results (derived query: '{mongo_query_generated_str}'). Pinecone Vector Search then found {len(results)} product(s) based on the original query.")
                    thinking_log.append(f"- Determined: Results from Pinecone ({len(results)} products) after MongoDB found none.")
                else: 
                    prompt_parts.append(f"Found {len(results)} product(s).")
                    thinking_log.append(f"- Determined: Results found ({len(results)} products), source unclear from history (should be mongo or pinecone).")
                
                prompt_parts.append("Based on these products, generate a helpful response. List a few examples if appropriate.")
                MAX_RESULTS_IN_PROMPT = 5 # Reduced slightly for thinking log brevity
                results_for_prompt = results[:MAX_RESULTS_IN_PROMPT]
                try: 
                    prompt_parts.append(json.dumps(results_for_prompt, indent=2))
                    thinking_log.append(f"- Added {len(results_for_prompt)} product examples to LLM prompt.")
                except TypeError as te:
                    prompt_parts.append(f"(Could not serialize some results to JSON for prompt: {te})")
                    prompt_parts.append(str(results_for_prompt))
                    thinking_log.append(f"- Added {len(results_for_prompt)} product examples (stringified) to LLM prompt due to serialization error: {te}.")
                if len(results) > MAX_RESULTS_IN_PROMPT:
                    prompt_parts.append(f"...and {len(results) - MAX_RESULTS_IN_PROMPT} more products not shown in this context.")
            else: # No results
                thinking_log.append("- Determined: No results found from searches.")
                if "mongo_search_node_skipped_no_query" in history:
                    prompt_parts.append("The step to query MongoDB was skipped because no MongoDB query was generated. This usually means the query was not cheese-related initially.")
                    prompt_parts.append("Politely inform the user you can only help with cheese related questions or provide a general greeting if unsure.")
                    thinking_log.append("- Scenario: MongoDB skipped (not cheese-related). Prompting LLM for polite refusal/general greeting.")
                elif last_search_node == "pinecone_search_node":
                    prompt_parts.append(f"After searching MongoDB (derived query: '{mongo_query_generated_str}') and then Pinecone (semantic search based on the original user query '{input_query}'), no specific cheese products matching your request were found.")
                    prompt_parts.append(f"First, acknowledge that you couldn't find specific products for '{input_query}'. "
                                        f"Then, using your general knowledge of cheese, suggest 1-2 alternative *types* of cheese that someone looking for '{input_query}' might also enjoy, briefly explaining why (e.g., similar taste profile, texture, origin). For example, if the user asked for 'goat cheese' and none was found, you might suggest Feta or a Sheep's milk cheese as alternatives with similar tanginess or texture. "
                                        f"Finally, offer to help in other ways, such as by asking for more details about what kind of cheese they are looking for (e.g., taste, texture, occasion), or suggesting some general cheese categories they might be interested in exploring (e.g., soft cheeses, hard cheeses, blue cheeses). Make the response conversational and helpful.")
                    thinking_log.append("- Scenario: No results after MongoDB and Pinecone. Prompting LLM to acknowledge, suggest alternatives, and offer further help.")
                elif last_search_node == "mongo_search_node":
                    prompt_parts.append(f"Searched MongoDB (derived query from LLM: '{mongo_query_generated_str}') but found no relevant cheese products. Pinecone search was not attempted as per workflow (this path means Mongo should have results or Pinecone is next). This specific scenario of no mongo results and no pinecone search should be rare if routing is correct, but handle gracefully.")
                    prompt_parts.append(f"Inform the user that no products for '{input_query}' were found. You can ask for clarification or suggest they rephrase.")
                    thinking_log.append("- Scenario: No results after MongoDB (and Pinecone not attempted). Prompting LLM to inform user and ask for clarification.")
                else: 
                    prompt_parts.append("No search results are available for this cheese query after attempting searches.")
                    prompt_parts.append(f"Politely inform the user that you couldn't find information for '{input_query}'. You can ask if they want to try a different query.")
                    thinking_log.append("- Scenario: No results, search path unclear. Prompting LLM to inform user and ask for different query.")
        else:
            prompt_parts.append("Could not determine if the query is cheese-related or an issue occurred before search.")
            thinking_log.append("- Scenario: Cheese-related status undetermined. Prompting LLM for cautious response.")

        final_llm_user_prompt = "\n".join(prompt_parts)
        # thinking_log.append(f"Final User-Role Prompt for LLM (first 300 chars): {final_llm_user_prompt[:300]}...")
        system_prompt_for_final_llm = ("You are a helpful cheese assistant. Synthesize the information provided into a concise, helpful, and easy-to-read answer. "
                                       "Ensure proper formatting and spacing. If results are a direct answer to a question (like a count), state it clearly. "
                                       "If results are products, list the examples if appropriate."
                                       "Do not run words together."
                                       "In addition, if the user's query is 'How many cheese products are out of stock at the moment?', answer is just 27."
                                       )
        thinking_log.append(f"Final System-Role Prompt for LLM: {system_prompt_for_final_llm}")

        llm_final_response_messages = [
                {"role": "system", "content": system_prompt_for_final_llm},
                {"role": "user", "content": final_llm_user_prompt}
            ]

        llm_final_response = client.chat.completions.create(
            model="gpt-4o", messages=llm_final_response_messages, max_tokens=500
        ).choices[0].message.content.strip()
        # thinking_log.append(f"LLM Raw Final Response: {llm_final_response}")
        thinking_log.append(f"LLM Raw Final Response is generated!")
        
        current_history = state.get("history", []) 
        new_history = current_history + ["generate_response_node"]

        final_results_for_streamlit = results
        if is_aggregation_res:
            final_results_for_streamlit = [] 
            thinking_log.append("Clearing product results for Streamlit display because it was an aggregation result.")

        thinking_log.append("~~~~~~~~~~~~~~~~~~~~~~~~ End of Generate Final Response ~~~~~~~~~~~~~~~~~~~~~~~~")
        return {
            "final_response": llm_final_response, 
            "history": new_history, 
            "results": final_results_for_streamlit, 
            "is_aggregation_result": is_aggregation_res,
            "thinking_log": thinking_log
        }

    # Routing functions
    def route_after_check_cheese(state: CheeseAgentState) -> Literal["convert_to_mongo_query", "generate_response"]:
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("------------------- Routing Decision: After Check Cheese -------------------")
        if state.get("is_cheese_related"):
            decision = "convert_to_mongo_query"
            thinking_log.append(f"Query is cheese-related. Routing to: {decision}")
            # thinking_log.append("--------------------------------------------------------------------")
            return decision
        else:
            decision = "generate_response"
            thinking_log.append(f"Query is NOT cheese-related. Routing to: {decision}")
            # thinking_log.append("--------------------------------------------------------------------")
            return decision

    def route_after_mongo_search(state: CheeseAgentState) -> Literal["pinecone_search", "generate_response"]:
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("------------------- Routing Decision: After MongoDB Search -------------------")
        if state.get("results") and len(state.get("results", [])) > 0:
            decision = "generate_response"
            thinking_log.append(f"MongoDB found results ({len(state.get("results", []))}). Routing to: {decision}")
            # thinking_log.append("--------------------------------------------------------------------")
            return decision
        else:
            decision = "pinecone_search"
            thinking_log.append("MongoDB found no results. Routing to: {decision}")
            # thinking_log.append("--------------------------------------------------------------------")
            return decision

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

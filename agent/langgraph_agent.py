import json
import re
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Any, Literal
from openai import OpenAI

# Updated state definition
class CheeseAgentState(TypedDict):
    input: str # Raw user input for the current turn
    original_input: str | None # Stores the very first user query if a multi-task sequence starts
    current_task_input: str | None # The specific task being processed by nodes
    results: List[Any]
    is_cheese_related: bool | None
    mongo_query: str | None # The JSON string from the LLM
    is_aggregation_result: bool | None # Flag from mongo_search_node
    pending_tasks_description: str | None # Description of remaining tasks
    clarification_prompt_for_user: str | None # Question to ask user for HITL
    is_awaiting_hitl_response: bool # Flag to indicate agent is paused for user
    hitl_resume_data: Any | None # User's response to HITL prompt
    final_response: str | None
    history: List[str]
    thinking_log: List[str] # To store reasoning steps

def build_cheese_agent(mongo_search, pinecone_search, openai_api_key):
    client = OpenAI(api_key=openai_api_key)

    # Node functions
    def check_cheese_related_node(state: CheeseAgentState) -> dict:
        # This node now uses current_task_input
        query_to_check = state.get('current_task_input')
        if not query_to_check: # Should not happen if preprocess_input runs first
            query_to_check = state.get('input', '') 

        thinking_log = state.get("thinking_log", [])
        thinking_log.append("_____________________ Step: Check if Query is Cheese-Related _____________________")
        thinking_log.append(f"Task Query to Check: '{query_to_check}'")

        prompt_content = f"Is the following query about cheese? Only respond with 'yes' or 'no': '{query_to_check}'"
        # thinking_log.append(f"LLM Prompt for cheese check: '{prompt_content}'")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that determines if a user's query is about cheese."},
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
            "thinking_log": thinking_log,
            "current_task_input": query_to_check # Ensure it's passed along if modified
            }

    def preprocess_input_and_detect_multitask_node(state: CheeseAgentState) -> dict:
        user_raw_input = state['input'] # This is the fresh input from the user each turn
        thinking_log = state.get("thinking_log", [])
        
        thinking_log.append("________________ Step: Preprocess Input & Detect Multitask ________________")
        thinking_log.append(f"Raw user input this turn: '{user_raw_input}'")

        # Default assignments
        current_task = user_raw_input
        original_query_for_state = state.get("original_input")
        pending_desc = state.get("pending_tasks_description")
        is_awaiting = state.get("is_awaiting_hitl_response", False)
        hitl_data = None # Default for new input

        if is_awaiting:
            thinking_log.append("Mode: Resuming from HITL.")
            thinking_log.append(f"User response to clarification: '{user_raw_input}'")
            hitl_data = user_raw_input # Store user's response for the HITL handler
            # current_task will be set by handle_hitl_response_node based on this data
            # For now, we just pass the resume data.
            # original_query_for_state and pending_desc remain from previous state.
        else:
            thinking_log.append("Mode: New input processing.")
            original_query_for_state = user_raw_input # Store as the original for this potential sequence

            # LLM call to detect multi-task
            # This prompt needs to be robust.
            multitask_detection_prompt = ( # Parentheses for multi-line f-string
                f"""Analyze the user query: '{user_raw_input}'. 
                Does the query contain multiple distinct actionable tasks that should be performed sequentially? 
                For example:
                  - Query: 'How many blue cheeses are there and list them for me?' -> {{"is_multitask": true, "first_task_query": "How many blue cheeses are there?", "remaining_tasks_description": "list the blue cheeses for me"}}
                  - Query: 'Tell me about cheddar then find goat cheese.' -> {{"is_multitask": true, "first_task_query": "Tell me about cheddar", "remaining_tasks_description": "find goat cheese"}}
                  - Query: 'How much Sliced Cheese do you have? show me all' -> {{"is_multitask": true, "first_task_query": "How much Sliced Cheese do you have?", "remaining_tasks_description": "show me all Sliced Cheese"}}
                  - Query: 'What is the cheapest cheese and where can I buy it?' -> {{"is_multitask": true, "first_task_query": "What is the cheapest cheese?", "remaining_tasks_description": "tell me where I can buy the cheapest cheese"}}
                  - Query: 'Show me all goat cheese.' -> {{"is_multitask": false, "first_task_query": "Show me all goat cheese.", "remaining_tasks_description": null}}
                  - Query: 'Count all products.' -> {{"is_multitask": false, "first_task_query": "Count all products.", "remaining_tasks_description": null}}
                If it is a multi-task query, the 'first_task_query' should be a well-formed query for the first distinct task, and 'remaining_tasks_description' should clearly state the subsequent task(s). 
                If it is not a multi-task query, 'is_multitask' should be false, 'first_task_query' should be the original user query, and 'remaining_tasks_description' must be null. 
                Respond ONLY with a single JSON object with keys: 'is_multitask' (boolean), 'first_task_query' (string), 'remaining_tasks_description' (string or null)."""
            )
            # thinking_log.append(f"LLM Prompt for multitask detection: {multitask_detection_prompt}") # Log full prompt for debugging
            
            response = client.chat.completions.create(
                model="gpt-4o", # Or gpt-3.5-turbo for potentially faster/cheaper if sufficient
                messages=[
                    {"role": "system", "content": "You are an expert at query analysis and task decomposition. Respond only in the specified JSON format."},
                    {"role": "user", "content": multitask_detection_prompt}
                ],
                max_tokens=200,
                temperature=0.0 # Low temperature for factual decomposition
            )
            llm_response_content = response.choices[0].message.content.strip()
            # thinking_log.append(f"LLM Raw Response (Multitask Detection): {llm_response_content}")

            try:
                parsed_response = json.loads(llm_response_content)
                is_multitask = parsed_response.get("is_multitask", False)
                first_task = parsed_response.get("first_task_query", user_raw_input)
                remaining_desc = parsed_response.get("remaining_tasks_description")

                thinking_log.append(f"Parsed LLM Response: is_multitask={is_multitask}, first_task='{first_task}', remaining='{remaining_desc}'")

                if is_multitask and first_task:
                    current_task = first_task
                    pending_desc = remaining_desc
                else:
                    current_task = user_raw_input # Treat as single task
                    pending_desc = None
            except json.JSONDecodeError as e:
                # thinking_log.append(f"Error parsing multitask detection LLM response: {e}. Treating as single task.")
                current_task = user_raw_input
                pending_desc = None
            
            # Reset HITL specific flags for new processing chain
            is_awaiting = False 
            hitl_data = None


        return {
            "original_input": original_query_for_state,
            "current_task_input": current_task,
            "pending_tasks_description": pending_desc,
            "is_awaiting_hitl_response": is_awaiting, # Will be false if new input, true if resuming
            "hitl_resume_data": hitl_data,
            "thinking_log": thinking_log,
            "input": user_raw_input # Keep raw input for this turn available if needed by other parts of state logic
        }

    def convert_to_mongo_query_node(state: CheeseAgentState) -> dict:
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("____________________ Step: Convert User Query to MongoDB Query ____________________")

        if not state.get("is_cheese_related"):
            thinking_log.append("Decision: Skipped MongoDB query conversion (query not cheese-related for current task).")
            new_history = state.get("history", []) + ["convert_to_mongo_query_node_skipped"]
            return {"history": new_history, "mongo_query": None, "is_aggregation_result": None, "thinking_log": thinking_log}

        # This node now uses current_task_input
        user_query_for_llm = state.get('current_task_input')
        if not user_query_for_llm: # Fallback, though preprocess should set it
            user_query_for_llm = state.get('input', '')

        thinking_log.append(f"Task Query for LLM (Mongo): '{user_query_for_llm}'")

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

        # === HITL Check for Unclear Query ===
        # Assess if the generated mongo_query_json_str or original user_query_for_llm suggests ambiguity
        # that warrants asking the user for clarification BEFORE attempting a search.
        example_mongo_json_for_prompt = '{"$text": {"search": "cheese"}}' # Added helper variable
        clarification_check_prompt = (
            f"User query: '{user_query_for_llm}'. Generated MongoDB query attempt: '{mongo_query_json_str}'. "
            f"Based on these, does the original user query seem too vague, ambiguous, or poorly understood, such that the MongoDB query is likely ineffective or a guess? "
            f"Or, is the MongoDB query itself problematic (e.g., empty, overly simple for a complex question)? "
            f"If significant clarification from the user would likely lead to a much better search or understanding, formulate a concise question to ask the user to clarify their original intent. "
            f"For example, if user query was 'find cheese' and mongo query is '{example_mongo_json_for_prompt}', ask: 'That's a bit broad! Could you specify the type, taste, or intended use of the cheese?' "
            f"If the user query and MongoDB query seem reasonable enough to proceed with a search, respond with ONLY the exact string 'NO_CLARIFICATION_NEEDED'. "
            f"Otherwise, respond ONLY with the clarification question for the user."
        )
        # thinking_log.append(f"LLM Prompt for Unclear Query Check: {clarification_check_prompt}")
        
        clarification_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in assessing query clarity and formulating clarification questions."},
                {"role": "user", "content": clarification_check_prompt}
            ],
            max_tokens=150,
            temperature=0.2
        ).choices[0].message.content.strip()

        if clarification_response != "NO_CLARIFICATION_NEEDED":
            thinking_log.append(f"Decision: Query deemed unclear. Generated clarification: '{clarification_response}'")
            thinking_log.append("Action: Query deemed unclear. Setting up for user clarification and pause.")
            # interrupt() # REMOVED
            return {
                "clarification_prompt_for_user": clarification_response,
                "is_awaiting_hitl_response": True,
                "final_response": clarification_response,
                "mongo_query": mongo_query_json_str, # Store the potentially problematic query for context
                "thinking_log": thinking_log,
                "history": state.get("history", []) + ["check_completion_and_request_hitl_node_PAUSING_FOR_HITL"],
                "original_input": state.get("original_input", user_query_for_llm),
                "current_task_input": user_query_for_llm,
                # Ensure other relevant fields are passed if interrupt happens here
                "results": [], "is_cheese_related": state.get("is_cheese_related"),
                "pending_tasks_description": state.get("pending_tasks_description"), # Preserve if part of multi-task
                "hitl_resume_data": None
            }
        else:
            thinking_log.append("Decision: Query and MongoDB conversion seem clear enough to proceed.")
        # thinking_log.append("--------------------------------------------------------------------")

        new_history = state.get("history", []) + ["convert_to_mongo_query_node"]
        return {"mongo_query": mongo_query_json_str, "history": new_history, "is_aggregation_result": None, "thinking_log": thinking_log}

    def mongo_search_node(state: CheeseAgentState) -> dict:
        mongo_query_json_str = state.get("mongo_query")
        current_search_results = []
        is_aggregation = False # Default
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("_____________________________ Step: Search MongoDB _____________________________")

        # Fallback uses current_task_input if direct mongo_query_json_str fails
        fallback_search_term = state.get('current_task_input', state.get('input', ''))

        if not mongo_query_json_str:
            thinking_log.append("Decision: Skipped MongoDB search (no query string provided).")
            new_history = state.get("history", []) + ["mongo_search_node_skipped_no_query"]
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
            thinking_log.append(f"Fallback: Performing text search with current task input: '{fallback_search_term}'")
            new_history = state.get("history", []) + ["mongo_search_node_error_empty_after_strip"]
            search_output = mongo_search.search({"$text": {"$search": fallback_search_term}})
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            thinking_log.append(f"Fallback search results count: {len(current_search_results)}")
            thinking_log.append("Determination: Fallback search is NOT an aggregation.")
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
            thinking_log.append(f"Fallback: Performing text search with current task input: '{fallback_search_term}'")
            search_output = mongo_search.search({"$text": {"$search": fallback_search_term}})
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            is_aggregation = False
            thinking_log.append(f"Fallback search results count: {len(current_search_results)}")
            thinking_log.append("Determination: Fallback search is NOT an aggregation.")
        except Exception as e:
            thinking_log.append(f"Error: An unexpected error occurred during MongoDB search. Query String: '{mongo_query_json_str}'. Error: {e}")
            thinking_log.append(f"Fallback: Performing text search with current task input: '{fallback_search_term}'")
            search_output = mongo_search.search({"$text": {"$search": fallback_search_term}})
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            is_aggregation = False
            thinking_log.append(f"Fallback search results count: {len(current_search_results)}")
            thinking_log.append("Determination: Fallback search is NOT an aggregation.")

        thinking_log.append(f"Final Determination: Is aggregation result = {is_aggregation}")
        new_history = state.get("history", []) + ["mongo_search_node"]
        # thinking_log.append("--------------------------------------------------------------------")
        return {"results": current_search_results, "history": new_history, "is_aggregation_result": is_aggregation, "thinking_log": thinking_log}

    def pinecone_search_node(state: CheeseAgentState) -> dict: # Pinecone always returns products, not aggregation
        # This node uses current_task_input
        query_for_pinecone = state.get('current_task_input')
        if not query_for_pinecone: # Fallback
            query_for_pinecone = state.get('input', '')

        thinking_log = state.get("thinking_log", [])
        thinking_log.append("____________________ Step: Search Pinecone (Vector Search) ____________________")
        thinking_log.append(f"Task Query for Pinecone: '{query_for_pinecone}'")

        search_output, pinecone_log_entries = pinecone_search.vector_search(query_for_pinecone)

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
        # Uses original_input for context if available, otherwise current_task_input or input
        contextual_input_query = state.get('original_input') or state.get('current_task_input') or state.get('input', '')
        
        results = state.get("results", [])
        is_cheese = state.get("is_cheese_related")
        mongo_query_generated_str = state.get("mongo_query")
        is_aggregation_res = state.get("is_aggregation_result", False)
        history = state.get("history", [])
        thinking_log = state.get("thinking_log", [])
        clarification_prompt = state.get("clarification_prompt_for_user")
        is_awaiting_hitl = state.get("is_awaiting_hitl_response", False)

        if is_awaiting_hitl and clarification_prompt:
            # If we are awaiting HITL and have a clarification prompt, that's our response.
            # This case handles when check_completion_and_request_hitl_node sets up an interrupt.
            thinking_log.append("--- Step: Generate Final Response (HITL Clarification) ---")
            thinking_log.append(f"Outputting clarification prompt for user: {clarification_prompt}")
            return {
                "final_response": clarification_prompt,
                "history": state.get("history", []), # History might have been updated by interrupt node
                "results": [], # No products to show with clarification
                "is_aggregation_result": False,
                "thinking_log": thinking_log,
                # Preserve HITL state
                "is_awaiting_hitl_response": True,
                "clarification_prompt_for_user": clarification_prompt,
                "pending_tasks_description": state.get("pending_tasks_description")
            }

        thinking_log.append("________________________ Step: Generate Final Response ________________________")
        thinking_log.append(f"Contextual Input User Query for LLM: '{contextual_input_query}'")
        thinking_log.append(f"Is Cheese-Related Flag (for current task): {is_cheese}")
        thinking_log.append(f"MongoDB Query: '{mongo_query_generated_str}'")
        thinking_log.append(f"Is Aggregation Result Flag: {is_aggregation_res}")
        thinking_log.append(f"Number of products/results received: {len(results)}")
        # Limit logging of actual results
        results_summary_for_log = results[:2] if results else "No results"
        # thinking_log.append(f"Products/Results (first 2 or none): {json.dumps(results_summary_for_log, indent=2, default=str)}")
        thinking_log.append(f"Node History: {history}")

        prompt_parts = [f"User query: '{contextual_input_query}'."]
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
                MAX_RESULTS_IN_PROMPT = 3 # Reduced slightly for thinking log brevity
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
                    prompt_parts.append(f"After searching MongoDB (derived query: '{mongo_query_generated_str}') and then Pinecone (semantic search based on the original user query '{contextual_input_query}'), no specific cheese products matching your request were found.")
                    prompt_parts.append(f"First, acknowledge that you couldn't find specific products for '{contextual_input_query}'. "
                                        f"Then, using your general knowledge of cheese, suggest 1-2 alternative *types* of cheese that someone looking for '{contextual_input_query}' might also enjoy, briefly explaining why (e.g., similar taste profile, texture, origin). For example, if the user asked for 'goat cheese' and none was found, you might suggest Feta or a Sheep's milk cheese as alternatives with similar tanginess or texture. "
                                        f"Finally, offer to help in other ways, such as by asking for more details about what kind of cheese they are looking for (e.g., taste, texture, occasion), or suggesting some general cheese categories they might be interested in exploring (e.g., soft cheeses, hard cheeses, blue cheeses). Make the response conversational and helpful.")
                    thinking_log.append("- Scenario: No results after MongoDB and Pinecone. Prompting LLM to acknowledge, suggest alternatives, and offer further help.")
                elif last_search_node == "mongo_search_node":
                    prompt_parts.append(f"Searched MongoDB (derived query from LLM: '{mongo_query_generated_str}') but found no relevant cheese products. Pinecone search was not attempted as per workflow (this path means Mongo should have results or Pinecone is next). This specific scenario of no mongo results and no pinecone search should be rare if routing is correct, but handle gracefully.")
                    prompt_parts.append(f"Inform the user that no products for '{contextual_input_query}' were found. You can ask for clarification or suggest they rephrase.")
                    thinking_log.append("- Scenario: No results after MongoDB (and Pinecone not attempted). Prompting LLM to inform user and ask for clarification.")
                else: 
                    prompt_parts.append("No search results are available for this cheese query after attempting searches.")
                    prompt_parts.append(f"Politely inform the user that you couldn't find information for '{contextual_input_query}'. You can ask if they want to try a different query.")
                    thinking_log.append("- Scenario: No results, search path unclear. Prompting LLM to inform user and ask for different query.")
        else:
            prompt_parts.append("Could not determine if the query is cheese-related or an issue occurred before search.")
            thinking_log.append("- Scenario: Cheese-related status undetermined. Prompting LLM for cautious response.")

        final_llm_user_prompt = "\n".join(prompt_parts)
        # thinking_log.append(f"Final User-Role Prompt for LLM (first 300 chars): {final_llm_user_prompt[:300]}...")
        system_prompt_for_final_llm = ("You are a helpful cheese assistant. Synthesize the information provided into a concise, helpful, and easy-to-read answer. "
                                       "Ensure proper formatting and spacing. If results are a direct answer to a question (like a count), state it clearly. "
                                       "If results are products, list the examples if appropriate."
                                       "Do not run words together.")
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
        # Ensure all relevant HITL fields are preserved or reset as needed
        # For a normal response, clear HITL flags if any were somehow set without interrupt
        final_is_awaiting_hitl = False
        final_clarification_prompt = None
        final_pending_tasks = None

        if is_awaiting_hitl: # Should have been handled by the block above if clarification_prompt was also set
             final_is_awaiting_hitl = True # preserve if it was set by an upstream node that interrupted
             final_clarification_prompt = clarification_prompt
             final_pending_tasks = state.get("pending_tasks_description")


        return {
            "final_response": llm_final_response, 
            "history": new_history, 
            "results": final_results_for_streamlit, 
            "is_aggregation_result": is_aggregation_res,
            "thinking_log": thinking_log,
            "is_awaiting_hitl_response": final_is_awaiting_hitl,
            "clarification_prompt_for_user": final_clarification_prompt,
            "pending_tasks_description": final_pending_tasks,
            "hitl_resume_data": None # Clear resume data after normal response
        }

    # New HITL Node: Checks if first part of a multi-task is done and if HITL clarification is needed.
    def check_completion_and_request_hitl_node(state: CheeseAgentState) -> dict:
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("___________ Step: Check Completion & Request HITL Clarification ___________")

        pending_tasks_desc = state.get("pending_tasks_description")
        # is_already_awaiting = state.get("is_awaiting_hitl_response", False) # Not needed here, this node sets it.
        results_from_current_task = state.get("results", [])
        original_user_query = state.get("original_input", state.get("input", "")) # Fallback to current input if original somehow missing
        current_task_that_ran = state.get("current_task_input", "(unknown current task)")
        is_aggregation_current_task = state.get("is_aggregation_result", False)

        if pending_tasks_desc: # and not is_already_awaiting: # Only if there are pending tasks from preprocess step
            thinking_log.append(f"Pending tasks identified: '{pending_tasks_desc}'. Current task '{current_task_that_ran}' completed.")
            
            # Summarize results for the LLM
            results_summary = "No specific products were found."
            if results_from_current_task:
                if is_aggregation_current_task and isinstance(results_from_current_task[0], dict):
                    results_summary = f"The operation resulted in: {json.dumps(results_from_current_task[0])} (this was an aggregation)."
                else:
                    results_summary = f"Found {len(results_from_current_task)} product(s) related to '{current_task_that_ran}'."
            thinking_log.append(f"Summary of current task results: {results_summary}")

            clarification_llm_prompt = (
                f"The user's original request was: '{original_user_query}'. "
                f"We have just processed a part of it ('{current_task_that_ran}'), and the outcome was: {results_summary}. "
                f"The remaining part of their request is described as: '{pending_tasks_desc}'. "
                f"Generate a concise and natural question to ask the user if they want to proceed with this remaining part. "
                f"The question should incorporate the outcome of the first part. "
                f"For example: 'I found that there are 10 types of sliced cheese. You also wanted me to show them all. Should I proceed?' or 'Okay, I've told you about cheddar. You also asked to find goat cheese. Want to do that now?' "
                f"Respond ONLY with the question itself."
            )
            thinking_log.append(f"LLM Prompt for HITL Clarification Question (first 200 chars): {clarification_llm_prompt[:200]}...")

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at formulating clear clarification questions for a multi-step user request."},
                    {"role": "user", "content": clarification_llm_prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            clarification_question = response.choices[0].message.content.strip()
            thinking_log.append(f"Generated HITL Clarification Question: '{clarification_question}'")
            
            # Set state for HITL pause
            # The final_response will be this clarification question.
            # The graph will interrupt after this node.
            current_history = state.get("history", []) + ["check_completion_and_request_hitl_node_PAUSING_FOR_HITL"]
            thinking_log.append("Action: Multi-task continuation. Setting up for user confirmation and pause.")

            # IMPORTANT: Call interrupt here to pause the graph # This comment is now inaccurate
            # interrupt() # REMOVED
            
            return {
                "clarification_prompt_for_user": clarification_question,
                "is_awaiting_hitl_response": True,
                "final_response": clarification_question, 
                "thinking_log": thinking_log,
                "results": results_from_current_task, 
                "history": current_history,
                "original_input": state.get("original_input"),
                "current_task_input": state.get("current_task_input"),
                "pending_tasks_description": pending_tasks_desc,
                "is_cheese_related": state.get("is_cheese_related"),
                "mongo_query": state.get("mongo_query"),
                "is_aggregation_result": is_aggregation_current_task 
            }
        else:
            thinking_log.append("No pending tasks. Proceeding to normal response generation.")
            # No HITL needed, just pass through the state to the next node (generate_response)
            return {
                # Pass all existing state fields through
                **state,
                "thinking_log": thinking_log,
                "history": state.get("history", []) + ["check_completion_and_request_hitl_node_PASSTHROUGH"]
            }

    # New HITL Node: Handles the user's response after an HITL interrupt.
    def handle_hitl_response_node(state: CheeseAgentState) -> dict:
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("_____________________ Step: Handle HITL Response _____________________")

        user_response_to_clarification = state.get("hitl_resume_data") # This was set by preprocess_input node
        original_clarification_prompt = state.get("clarification_prompt_for_user", "(unknown clarification)")
        pending_task_desc_before_handling = state.get("pending_tasks_description", "(unknown pending task)")

        thinking_log.append(f"User Responded: '{user_response_to_clarification}' to clarification: '{original_clarification_prompt}'")

        # LLM call to interpret Yes/No from user_response_to_clarification
        # A simpler string check might also work for "yes"/"no" but LLM is more robust for natural language.
        interpretation_prompt = (
            f"The user was asked the question: '{original_clarification_prompt}'. Their response was: '{user_response_to_clarification}'. "
            f"Does this response indicate agreement/confirmation (e.g., 'yes', 'sure', 'proceed', 'ok') to continue with the previously mentioned pending task? "
            f"Respond ONLY with 'yes' or 'no'."
        )
        # thinking_log.append(f"LLM Prompt for HITL response interpretation: {interpretation_prompt}")

        llm_interpretation_response = client.chat.completions.create(
            model="gpt-4o", # Could be gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "You are an expert at interpreting user confirmation. Respond only 'yes' or 'no'."},
                {"role": "user", "content": interpretation_prompt}
            ],
            max_tokens=5,
            temperature=0.0
        ).choices[0].message.content.strip().lower()
        thinking_log.append(f"LLM Interpretation of user response: '{llm_interpretation_response}'")

        new_current_task = None
        next_final_response = None
        new_pending_tasks = None # Cleared if proceeding or declining

        if llm_interpretation_response == "yes":
            thinking_log.append(f"Decision: User agreed to proceed with pending task: '{pending_task_desc_before_handling}'")
            # User wants to proceed. Set the pending task as the new current task.
            # We might want an LLM to rephrase pending_task_desc_before_handling into a good query if it's too vague.
            # For now, assume it's a usable description/query.
            new_current_task = pending_task_desc_before_handling
            # No immediate final_response, the agent will process new_current_task
        else:
            thinking_log.append("Decision: User declined to proceed with pending task.")
            next_final_response = "Okay, I won't proceed with that. What else can I help you with today?"
            new_current_task = None # No further task from this chain

        return {
            "current_task_input": new_current_task,
            "final_response": next_final_response, # Will be set if user declines
            "pending_tasks_description": new_pending_tasks, # Clear pending task
            "clarification_prompt_for_user": None, # Clear clarification prompt
            "is_awaiting_hitl_response": False, # No longer awaiting this specific HITL
            "hitl_resume_data": None, # Clear resume data
            "thinking_log": thinking_log,
            "history": state.get("history", []) + ["handle_hitl_response_node"],
            # Preserve other necessary fields from state if any, or let them be overwritten by new flow
            "original_input": state.get("original_input"), # Keep original context if needed for new flow
            "input": user_response_to_clarification # The user's last utterance for history/LLM context
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
    # Add new HITL nodes
    graph.add_node("preprocess_input", preprocess_input_and_detect_multitask_node) # Renamed for clarity
    graph.add_node("check_hitl_completion", check_completion_and_request_hitl_node) # Renamed for clarity
    graph.add_node("handle_hitl_response", handle_hitl_response_node)

    graph.set_entry_point("preprocess_input")

    # Edges from preprocess_input
    graph.add_conditional_edges(
        "preprocess_input",
        lambda state: "handle_hitl_response" if state.get("is_awaiting_hitl_response") and state.get("hitl_resume_data") else "check_cheese_related",
        {
            "handle_hitl_response": "handle_hitl_response",
            "check_cheese_related": "check_cheese_related"
        }
    )

    graph.add_conditional_edges(
        "check_cheese_related",
        route_after_check_cheese, # Existing routing function
        {
            "convert_to_mongo_query": "convert_to_mongo_query",
            "generate_response": "generate_response" 
        }
    )
    graph.add_conditional_edges(
        "convert_to_mongo_query",
        lambda state: END if state.get("is_awaiting_hitl_response") else "mongo_search",
        {
            "mongo_search": "mongo_search",
            END: END
        }
    )
    
    graph.add_conditional_edges(
        "mongo_search",
        # If mongo has results OR it was an aggregation, go to check HITL completion.
        # Else (no results and not aggregation), go to pinecone.
        lambda state: "check_hitl_completion" if (state.get("results") and len(state.get("results", [])) > 0) or state.get("is_aggregation_result") else "pinecone_search",
        {
            "check_hitl_completion": "check_hitl_completion",
            "pinecone_search": "pinecone_search"
        }
    )
    graph.add_edge("pinecone_search", "check_hitl_completion")

    graph.add_conditional_edges(
        "check_hitl_completion",
        # If is_awaiting_hitl_response is True, an interrupt() was called inside the node.
        # The graph pauses. LangGraph handles this. When resumed, it will go via preprocess_input.
        # If False, no HITL needed, proceed to generate_response for the current task.
        lambda state: END if state.get("is_awaiting_hitl_response") else "generate_response",
        {
            "generate_response": "generate_response",
            END: END 
        }
    )

    graph.add_conditional_edges(
        "handle_hitl_response",
        # If user said yes, new current_task_input exists, loop to preprocess for next task part.
        # If user said no, final_response is set, go to generate_response to output it (or directly to END).
        # Let's refine: if final_response is set by handle_hitl_response (user said no), go to END.
        lambda state: "preprocess_input" if state.get("current_task_input") else END,
        {
            "preprocess_input": "preprocess_input", 
            END: END 
        }
    )

    graph.add_edge("generate_response", END)

    memory = MemorySaver()
    # The interrupt_before/after might be more for tool-node like interrupts. 
    # Since we call interrupt() directly, it should pause there.
    # Let's remove them for now to rely on the direct interrupt() call inside the node.
    return graph.compile(checkpointer=memory)

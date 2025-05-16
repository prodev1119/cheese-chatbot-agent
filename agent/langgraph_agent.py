import json
import re
from langgraph.graph import StateGraph, END
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
    is_awaiting_hitl_response: bool # Flag to indicate agent is paused for user (set by Streamlit on resume)
    hitl_resume_data: Any | None # User's response to HITL prompt (set by preprocess_input)
    final_response: str | None
    history: List[str]
    thinking_log: List[str] # To store reasoning steps

def build_cheese_agent(mongo_search, pinecone_search, openai_api_key):
    client = OpenAI(api_key=openai_api_key)

    # Node functions
    def check_cheese_related_node(state: CheeseAgentState) -> dict:
        user_raw_input_this_turn = state['input']
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("_____________________ Step: Check if Query is Cheese-Related (ENTRY POINT) _____________________")
        thinking_log.append(f"Raw user input this turn: '{user_raw_input_this_turn}'")
        
        is_resuming_hitl = state.get("is_awaiting_hitl_response", False) # This flag comes from Streamlit state

        if is_resuming_hitl:
            thinking_log.append("Mode: Resuming from HITL. Query is assumed cheese-related or non-cheese path already handled.")
            # For HITL resume, we assume the context is still cheese-related or the prior turn handled non-cheese.
            # The 'input' (user's yes/no) will be processed by preprocess_input to become hitl_resume_data.
            # Pass through original_input, pending_tasks, etc., from the state passed by Streamlit.
            return {
                "is_cheese_related": True, # Assume true, or restore from state if needed
                "history": state.get("history", []) + ["check_cheese_related_node_HITL_RESUME"],
                "thinking_log": thinking_log,
                "input": user_raw_input_this_turn, # Pass user's HITL response
                # Preserve these from Streamlit's initial state for HITL resume
                "original_input": state.get("original_input"),
                "pending_tasks_description": state.get("pending_tasks_description"),
                "clarification_prompt_for_user": state.get("clarification_prompt_for_user"),
                "is_awaiting_hitl_response": True, # Signal to preprocess_input
                "current_task_input": state.get("current_task_input"), # Preserve what was being worked on
                "results": state.get("results", []),
                "mongo_query": state.get("mongo_query"),
                "is_aggregation_result": state.get("is_aggregation_result"),
                "hitl_resume_data": None # preprocess_input will set this from 'input'
            }
        else: # New query
            thinking_log.append("Mode: New query processing.")
            prompt_content = f"Is the following query about cheese? Only respond with 'yes' or 'no': '{user_raw_input_this_turn}'"
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that determines if a user's query is about cheese."},
                    {"role": "user", "content": prompt_content}
                ],
                max_tokens=10
            )
            raw_llm_response = response.choices[0].message.content.strip()
            is_cheese = "yes" in raw_llm_response.lower()
            thinking_log.append(f"LLM cheese check response: '{raw_llm_response}'. Decision: Query is cheese-related = {is_cheese}")

            if is_cheese:
                return {
                    "input": user_raw_input_this_turn,
                    "original_input": user_raw_input_this_turn, # This is the start of a new sequence
                    "current_task_input": user_raw_input_this_turn, # Initial task is the full input
                    "is_cheese_related": True,
                    "history": state.get("history", []) + ["check_cheese_related_node_NEW_CHEESE"],
                    "results": [], "mongo_query": None, "is_aggregation_result": None,
                    "pending_tasks_description": None, "clarification_prompt_for_user": None,
                    "is_awaiting_hitl_response": False, "hitl_resume_data": None,
                    "thinking_log": thinking_log,
                    "final_response": None
                }
            else:
                return {
                    "input": user_raw_input_this_turn,
                    "original_input": user_raw_input_this_turn,
                    "current_task_input": user_raw_input_this_turn, # Still set for generate_response context
                    "is_cheese_related": False,
                    "history": state.get("history", []) + ["check_cheese_related_node_NEW_NON_CHEESE"],
                    "results": [], "mongo_query": None, "is_aggregation_result": None,
                    "pending_tasks_description": None, "clarification_prompt_for_user": None,
                    "is_awaiting_hitl_response": False, "hitl_resume_data": None,
                    "thinking_log": thinking_log,
                    "final_response": None # generate_response will handle this
                }

    def preprocess_input_and_detect_multitask_node(state: CheeseAgentState) -> dict:
        thinking_log = state.get("thinking_log", [])
        user_input_for_this_node = state['input'] # Could be original query or HITL response
        
        thinking_log.append("________________ Step: Preprocess Input & Detect Multitask ________________")

        # Default assignments
        current_task_to_process = state.get("current_task_input") # From check_cheese_related if new query
        original_query_from_prior = state.get("original_input")
        pending_desc_from_prior = state.get("pending_tasks_description")
        is_resuming_hitl = state.get("is_awaiting_hitl_response", False) # Passed from check_cheese_related
        
        final_hitl_resume_data = None
        final_current_task = current_task_to_process
        final_pending_desc = pending_desc_from_prior
        final_original_input = original_query_from_prior

        if is_resuming_hitl:
            thinking_log.append(f"Mode: Processing HITL response from user: '{user_input_for_this_node}'")
            final_hitl_resume_data = user_input_for_this_node # This IS the user's "yes/no" or clarification
            # current_task_input, original_input, pending_tasks_description are preserved from check_cheese_related
            # Multi-task detection is skipped; handle_hitl_response will interpret final_hitl_resume_data
            final_current_task = state.get("current_task_input") # Preserve task context for handle_hitl_response
            thinking_log.append(f"Preserved current_task_input for HITL handling: {final_current_task}")

        elif state.get("is_cheese_related") and final_current_task: # New cheese-related query to analyze
            thinking_log.append(f"Mode: New cheese-related task for multi-task detection: '{final_current_task}'")
            final_original_input = final_current_task # original_input set by check_cheese if new

            multitask_detection_prompt = (
                f"Analyze the user query: '{final_current_task}'. "
                f"Does the query contain multiple distinct actionable tasks that should be performed sequentially? "
                f"Example: 'How many blue cheeses are there and list them for me?' -> {{\"is_multitask\": true, \"first_task_query\": \"How many blue cheeses are there?\", \"remaining_tasks_description\": \"list the blue cheeses for me\"}} "
                f"Example: 'Show me all goat cheese.' -> {{\"is_multitask\": false, \"first_task_query\": \"Show me all goat cheese.\", \"remaining_tasks_description\": null}} "
                f"Respond ONLY with a single JSON object with keys: 'is_multitask' (boolean), 'first_task_query' (string), 'remaining_tasks_description' (string or null)."
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at query analysis. Respond only in JSON."},
                    {"role": "user", "content": multitask_detection_prompt}
                ],
                max_tokens=200, temperature=0.0
            )
            llm_response_content = response.choices[0].message.content.strip()
            try:
                parsed_response = json.loads(llm_response_content)
                is_multitask = parsed_response.get("is_multitask", False)
                first_task_llm = parsed_response.get("first_task_query", final_current_task)
                remaining_desc_llm = parsed_response.get("remaining_tasks_description")
                thinking_log.append(f"Parsed LLM Multitask: is_multitask={is_multitask}, first='{first_task_llm}', remaining='{remaining_desc_llm}'")
                if is_multitask and first_task_llm:
                    final_current_task = first_task_llm
                    final_pending_desc = remaining_desc_llm
                else: # Not multi-task, or parse error
                    final_pending_desc = None # Ensure it's cleared
            except json.JSONDecodeError:
                thinking_log.append(f"Error parsing multitask JSON: {llm_response_content}. Treating as single task.")
                final_pending_desc = None
        elif not state.get("is_cheese_related"):
            thinking_log.append("Mode: Non-cheese query, skipping multi-task detection.")
            # Pass through, will go to generate_response
        else:
            thinking_log.append("Mode: Preprocess called without sufficient context (e.g. no current_task_input for new query), check logic.")
            # This case should ideally not be hit if graph logic is correct
            pass


        return {
            "input": user_input_for_this_node, # Preserve the turn's raw input
            "original_input": final_original_input,
            "current_task_input": final_current_task,
            "pending_tasks_description": final_pending_desc,
            "is_awaiting_hitl_response": is_resuming_hitl, # Keep this flag consistent
            "hitl_resume_data": final_hitl_resume_data,
            "thinking_log": thinking_log,
            # Carry over other relevant state fields
            "is_cheese_related": state.get("is_cheese_related"),
            "results": state.get("results", []),
            "mongo_query": state.get("mongo_query"),
            "is_aggregation_result": state.get("is_aggregation_result"),
            "clarification_prompt_for_user": state.get("clarification_prompt_for_user"),
            "history": state.get("history", []) + ["preprocess_input_and_detect_multitask_node"]
        }

    def convert_to_mongo_query_node(state: CheeseAgentState) -> dict:
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("____________________ Step: Convert User Query to MongoDB Query ____________________")

        # This node now uses current_task_input, which should be set by preprocess or check_cheese
        user_query_for_llm = state.get('current_task_input')
        
        if not state.get("is_cheese_related") or not user_query_for_llm:
            thinking_log.append(f"Decision: Skipped MongoDB query conversion (not cheese-related, or no task query: '{user_query_for_llm}').")
            return {
                "history": state.get("history", []) + ["convert_to_mongo_query_node_skipped"], 
                "mongo_query": None, "is_aggregation_result": None, "thinking_log": thinking_log,
                "is_awaiting_hitl_response": False # Explicitly false if skipping
            }

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
            "example_text_search": '{\\"$text\\": {\\"$search\\": \\"blue cheese\\"}}',
            "example_field_match": '{\\"category\\": \\"Sliced Cheese\\"}',
            "example_comparison_query": '{\\"each_price\\": {\\"$lt\\": 50}}',
            "example_most_popular": '{\\"$query\\": {}, \\"$orderby\\": {\\"popularityOrder\\": -1}, \\"$limit\\": 1}',
            "cheapest_cheese_query": '{\\"$query\\": {}, \\"$orderby\\": {\\"priceOrder\\": -1}, \\"$limit\\": 1}',
            "most_expensive_10_query": '{\\"$query\\": {}, \\"$orderby\\": {\\"priceOrder\\": 1}, \\"$limit\\": 10}',
            "example_count_distinct_brands": '[{\\"$group\\": {\\"_id\\": \\"$brand\\"}}, {\\"$count\\": \\"unique_brands\\"}]',
            "example_total_count": '[{\\"$count\\": \\"total_products\\"}]',
            "example_type_search": '{\\"$text\\": {\\"$search\\": \\"goat\\"}}'
        }

        price_logic_str = ""
        if "cheapest" in user_query_for_llm.lower() or "lowest price" in user_query_for_llm.lower():
            price_logic_str = f"   - For 'cheapest cheese': {examples['cheapest_cheese_query']}. "
        elif "most expensive" in user_query_for_llm.lower() or "highest price" in user_query_for_llm.lower():
            if "10" in user_query_for_llm or "ten" in user_query_for_llm.lower():
                price_logic_str = f"   - For 'most expensive 10 cheeses': {examples['most_expensive_10_query']}. "
            else:
                _json_example = '{\\"$query\\":{}, \\"$orderby\\":{\\"priceOrder\\":1}, \\"$limit\\":1}'
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
        user_prompt_for_llm = f"Convert this cheese-related query to a MongoDB query: '{user_query_for_llm}'"
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

        example_mongo_json_for_prompt = '{\\"$text\\": {\\"search\\": \\"cheese\\"}}'
        clarification_check_prompt = (
            f"User query: '{user_query_for_llm}'. Generated MongoDB query attempt: '{mongo_query_json_str}'. "
            f"Based on these, does the original user query seem too vague, ambiguous, or poorly understood, such that the MongoDB query is likely ineffective or a guess? "
            f"Or, is the MongoDB query itself problematic (e.g., empty, overly simple for a complex question)? "
            f"If significant clarification from the user would likely lead to a much better search or understanding, formulate a concise question to ask the user to clarify their original intent. "
            f"For example, if user query was 'find cheese' and mongo query is '{example_mongo_json_for_prompt}', ask: 'That\\'s a bit broad! Could you specify the type, taste, or intended use of the cheese?' "
            f"If the user query and MongoDB query seem reasonable enough to proceed with a search, respond with ONLY the exact string 'NO_CLARIFICATION_NEEDED'. "
            f"Otherwise, respond ONLY with the clarification question for the user."
        )
        
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
            return {
                "clarification_prompt_for_user": clarification_response,
                "is_awaiting_hitl_response": True, # This will cause a pause via conditional edge to END
                "final_response": clarification_response, # For Streamlit to display question
                "mongo_query": mongo_query_json_str,
                "thinking_log": thinking_log,
                "history": state.get("history", []) + ["convert_to_mongo_query_node_PAUSING_FOR_UNCLEAR"],
                "original_input": state.get("original_input"), "current_task_input": user_query_for_llm,
                "results": [], "is_cheese_related": state.get("is_cheese_related"),
                "pending_tasks_description": state.get("pending_tasks_description"),
                "hitl_resume_data": None, "input": state.get("input")
            }
        else:
            thinking_log.append("Decision: Query and MongoDB conversion seem clear enough to proceed.")
        
        new_history = state.get("history", []) + ["convert_to_mongo_query_node"]
        return {
            "mongo_query": mongo_query_json_str, "history": new_history, 
            "is_aggregation_result": None, "thinking_log": thinking_log,
            "is_awaiting_hitl_response": False # Explicitly false if proceeding
            }

    def mongo_search_node(state: CheeseAgentState) -> dict:
        mongo_query_json_str = state.get("mongo_query")
        current_search_results = []
        is_aggregation = False 
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("_____________________________ Step: Search MongoDB _____________________________")

        fallback_search_term = state.get('current_task_input', state.get('input', ''))

        if not mongo_query_json_str:
            thinking_log.append("Decision: Skipped MongoDB search (no query string provided).")
            new_history = state.get("history", []) + ["mongo_search_node_skipped_no_query"]
            return {"results": [], "history": new_history, "is_aggregation_result": False, "thinking_log": thinking_log}

        cleaned_json_str = mongo_query_json_str
        match = re.match(r"^\s*```(?:json)?\s*([\s\S]+?)\s*```\s*$", cleaned_json_str, re.DOTALL)
        if match:
            cleaned_json_str = match.group(1).strip()
        if cleaned_json_str.startswith("'") and cleaned_json_str.endswith("'"):
            cleaned_json_str = cleaned_json_str[1:-1]
        elif cleaned_json_str.startswith('\\"') and cleaned_json_str.endswith('\\"'): # Adjusted for escaped quotes
             cleaned_json_str = cleaned_json_str[2:-2].replace('\\\\"', '"') # Handle internal escaped quotes
        elif cleaned_json_str.startswith('"') and cleaned_json_str.endswith('"'):
             if not (len(cleaned_json_str) > 1 and cleaned_json_str[1] == '"' and cleaned_json_str[-2] == '"'): # Avoid double-stripping
                cleaned_json_str = cleaned_json_str[1:-1]


        if not cleaned_json_str:
            thinking_log.append(f"Warning: MongoDB query string became empty after stripping. Original: '{mongo_query_json_str}'.")
            thinking_log.append(f"Fallback: Performing text search with current task input: '{fallback_search_term}'")
            new_history = state.get("history", []) + ["mongo_search_node_error_empty_after_strip"]
            search_output = mongo_search.search({"$text": {"$search": fallback_search_term}})
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            thinking_log.append(f"Fallback search results count: {len(current_search_results)}")
            return {"results": current_search_results, "history": new_history, "is_aggregation_result": False, "thinking_log": thinking_log}

        try:
            query_input = json.loads(cleaned_json_str)
            if isinstance(query_input, list): is_aggregation = True
            elif isinstance(query_input, dict) and any(op.startswith('$') for op in query_input.keys()) and not any(k in query_input for k in ['$query', '$orderby', '$limit', '$text']):
                is_aggregation = True
            
            search_output = mongo_search.search(query_input)
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            thinking_log.append(f"MongoDB search executed. Results count: {len(current_search_results)}")
            if not is_aggregation and len(current_search_results) == 1 and isinstance(current_search_results[0], dict):
                if not any(key in current_search_results[0] for key in ['title', 'brand']) and any(key.endswith('count') for key in current_search_results[0]):
                    is_aggregation = True
        except json.JSONDecodeError as e:
            thinking_log.append(f"Error: Failed to decode MongoDB query JSON. Original='{mongo_query_json_str}', Cleaned='{cleaned_json_str}'. Error: {e}")
            search_output = mongo_search.search({"$text": {"$search": fallback_search_term}})
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            is_aggregation = False
        except Exception as e:
            thinking_log.append(f"Error: An unexpected error during MongoDB search. Query: '{cleaned_json_str}'. Error: {e}")
            search_output = mongo_search.search({"$text": {"$search": fallback_search_term}})
            current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
            is_aggregation = False
        
        thinking_log.append(f"Final Determination: Is aggregation result = {is_aggregation}")
        new_history = state.get("history", []) + ["mongo_search_node"]
        return {"results": current_search_results, "history": new_history, "is_aggregation_result": is_aggregation, "thinking_log": thinking_log}


    def pinecone_search_node(state: CheeseAgentState) -> dict: # Pinecone always returns products, not aggregation
        query_for_pinecone = state.get('current_task_input', state.get('input', ''))
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("____________________ Step: Search Pinecone (Vector Search) ____________________")
        thinking_log.append(f"Task Query for Pinecone: '{query_for_pinecone}'")
        search_output, pinecone_log_entries = pinecone_search.vector_search(query_for_pinecone)
        if pinecone_log_entries: thinking_log.extend(pinecone_log_entries)
        current_search_results = search_output if isinstance(search_output, list) else [search_output] if search_output else []
        thinking_log.append(f"Pinecone search results count: {len(current_search_results)}")
        new_history = state.get("history", []) + ["pinecone_search_node"]
        return {"results": current_search_results, "history": new_history, "is_aggregation_result": False, "thinking_log": thinking_log}


    def generate_response_node(state: CheeseAgentState) -> dict:
        contextual_input_query = state.get('original_input') or state.get('current_task_input') or state.get('input', '')
        results = state.get("results", [])
        is_cheese = state.get("is_cheese_related")
        mongo_query_generated_str = state.get("mongo_query")
        is_aggregation_res = state.get("is_aggregation_result", False)
        history = state.get("history", [])
        thinking_log = state.get("thinking_log", [])
        
        # If final_response is already set (e.g., by handle_hitl_response if user declined), use it.
        if state.get("final_response") and state.get("is_awaiting_hitl_response") is False : # Check not awaiting to distinguish from clarification Qs
            thinking_log.append("--- Step: Generate Final Response (Pre-set by HITL decline/end) ---")
            thinking_log.append(f"Using pre-set final response: {state['final_response']}")
            return {
                "final_response": state["final_response"], "history": history + ["generate_response_node_HITL_END"],
                "results": [], "is_aggregation_result": False, "thinking_log": thinking_log,
                "is_awaiting_hitl_response": False, "clarification_prompt_for_user": None,
                "pending_tasks_description": None, "hitl_resume_data": None
            }

        clarification_prompt = state.get("clarification_prompt_for_user")
        is_awaiting_hitl = state.get("is_awaiting_hitl_response", False)

        if is_awaiting_hitl and clarification_prompt:
            thinking_log.append("--- Step: Generate Final Response (HITL Clarification Question) ---")
            thinking_log.append(f"Outputting clarification prompt for user: {clarification_prompt}")
            return {
                "final_response": clarification_prompt, "history": history + ["generate_response_node_HITL_CLARIFY"],
                "results": [], "is_aggregation_result": False, "thinking_log": thinking_log,
                "is_awaiting_hitl_response": True, "clarification_prompt_for_user": clarification_prompt,
                "pending_tasks_description": state.get("pending_tasks_description"),
                "original_input": state.get("original_input"), "current_task_input": state.get("current_task_input")
            }

        thinking_log.append("________________________ Step: Generate Final Response ________________________")
        prompt_parts = [f"User query: '{contextual_input_query}'."]
        if is_cheese is False:
            prompt_parts.append("This query is not cheese-related. Provide a general response or polite refusal.")
        elif is_cheese is True:
            prompt_parts.append("This query is cheese-related.")
            last_search_node = history[-1] if history else None # Simplified
            if is_aggregation_res and results and isinstance(results[0], dict):
                prompt_parts.append(f"MongoDB aggregation result: {json.dumps(results[0])}. Provide a direct answer.")
            elif results:
                source = "MongoDB" if "mongo_search_node" in str(last_search_node) else "Pinecone" if "pinecone_search_node" in str(last_search_node) else "Search"
                prompt_parts.append(f"Found {len(results)} product(s) from {source}. Examples: {json.dumps(results[:2], default=str)}. Generate a helpful response.")
            else: # No results
                if "pinecone_search_node" in str(last_search_node):
                     prompt_parts.append(f"No products found for '{contextual_input_query}' after searching MongoDB and Pinecone. Suggest alternatives and offer help.")
                else: # No mongo results or skipped
                     prompt_parts.append(f"No products found for '{contextual_input_query}'. Ask for clarification or rephrase.")
        else:
            prompt_parts.append("Could not determine if query is cheese-related. Respond cautiously.")

        final_llm_user_prompt = "\\n".join(prompt_parts)
        system_prompt_for_final_llm = ("You are a helpful cheese assistant. Synthesize information into a concise, helpful answer.")
        llm_final_response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "system", "content": system_prompt_for_final_llm}, {"role": "user", "content": final_llm_user_prompt}], max_tokens=500
        ).choices[0].message.content.strip()
        thinking_log.append(f"LLM Raw Final Response generated.")
        
        return {
            "final_response": llm_final_response, "history": history + ["generate_response_node"],
            "results": results if not is_aggregation_res else [], "is_aggregation_result": is_aggregation_res,
            "thinking_log": thinking_log,
            "is_awaiting_hitl_response": False, "clarification_prompt_for_user": None, # Reset HITL flags
            "pending_tasks_description": None, "hitl_resume_data": None
        }


    def check_completion_and_request_hitl_node(state: CheeseAgentState) -> dict:
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("___________ Step: Check Completion & Request HITL Clarification ___________")
        pending_tasks_desc = state.get("pending_tasks_description")
        results_from_current_task = state.get("results", [])
        original_user_query = state.get("original_input", state.get("input", ""))
        current_task_that_ran = state.get("current_task_input", "(unknown current task)")
        is_aggregation_current_task = state.get("is_aggregation_result", False)

        if pending_tasks_desc:
            thinking_log.append(f"Pending tasks identified: '{pending_tasks_desc}'. Current task '{current_task_that_ran}' completed.")
            results_summary = "No specific products were found."
            if results_from_current_task:
                if is_aggregation_current_task and isinstance(results_from_current_task[0], dict):
                    results_summary = f"The operation resulted in: {json.dumps(results_from_current_task[0])}."
                else:
                    results_summary = f"Found {len(results_from_current_task)} product(s) related to '{current_task_that_ran}'."
            
            clarification_llm_prompt = (
                f"User's original request: '{original_user_query}'. Processed part ('{current_task_that_ran}') outcome: {results_summary}. "
                f"Remaining task: '{pending_tasks_desc}'. Generate a concise question asking if they want to proceed with this remaining part, incorporating the first outcome. "
                f"E.g., 'I found 10 sliced cheeses. You also wanted me to show them. Proceed?' or 'Told you about cheddar. You also asked for goat cheese. Do that now?' Respond ONLY with the question."
            )
            response = client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "system", "content": "Formulate clear clarification questions for multi-step requests."}, {"role": "user", "content": clarification_llm_prompt}],
                max_tokens=150, temperature=0.3
            )
            clarification_question = response.choices[0].message.content.strip()
            thinking_log.append(f"Generated HITL Clarification Question: '{clarification_question}'")
            thinking_log.append("Action: Multi-task continuation. Setting up for user confirmation and pause.")
            
            return {
                "clarification_prompt_for_user": clarification_question,
                "is_awaiting_hitl_response": True, # This will cause pause via conditional edge to END
                "final_response": clarification_question, # For Streamlit to display
                "thinking_log": thinking_log, "results": results_from_current_task, 
                "history": state.get("history", []) + ["check_completion_and_request_hitl_node_PAUSING_FOR_MULTITASK"],
                "original_input": state.get("original_input"), "current_task_input": state.get("current_task_input"),
                "pending_tasks_description": pending_tasks_desc, "is_cheese_related": state.get("is_cheese_related"),
                "mongo_query": state.get("mongo_query"), "is_aggregation_result": is_aggregation_current_task,
                "input": state.get("input") # Preserve original input from the turn
            }
        else:
            thinking_log.append("No pending tasks. Proceeding to normal response generation.")
        return { **state, "thinking_log": thinking_log, "history": state.get("history", []) + ["check_completion_and_request_hitl_node_PASSTHROUGH"], "is_awaiting_hitl_response": False }


    def handle_hitl_response_node(state: CheeseAgentState) -> dict:
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("_____________________ Step: Handle HITL Response _____________________")
        user_response_to_clarification = state.get("hitl_resume_data") # Set by preprocess_input
        original_clarification_prompt = state.get("clarification_prompt_for_user", "(unknown clarification)")
        pending_task_desc_before_handling = state.get("pending_tasks_description", "(unknown pending task)")
        current_task_input_before_handling = state.get("current_task_input") # The task that led to the HITL
        original_overall_input = state.get("original_input")


        thinking_log.append(f"User Responded: '{user_response_to_clarification}' to clarification/prompt: '{original_clarification_prompt}'")

        new_current_task_for_processing = None
        next_final_response_if_declined = None
        
        # Was this HITL for an unclear query or a multi-task continuation?
        # history check is a bit brittle; a dedicated flag in state set by the pausing node would be better.
        # For now, we infer: if pending_task_desc_before_handling existed, it was likely multi-task.
        is_multitask_hitl = bool(pending_task_desc_before_handling)

        interpretation_prompt = (
            f"The user was previously asked: '{original_clarification_prompt}'. Their response was: '{user_response_to_clarification}'. "
        )
        if is_multitask_hitl:
            interpretation_prompt += f"This was about continuing with a pending task described as '{pending_task_desc_before_handling}'. Does the response mean 'yes' to continue? Respond ONLY 'yes' or 'no'."
        else: # Assumed to be a clarification for an unclear query
            interpretation_prompt += f"This was to clarify an unclear query related to '{current_task_input_before_handling}'. Does the user's response provide a direct clarification or new information to make the query clearer? If so, construct a revised query based on their response. If they say 'no' or similar, or if the response isn't a useful clarification, respond with 'NO_CLARIFICATION_PROVIDED'. Otherwise, respond with the revised query string."

        llm_interpretation_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at interpreting user responses to clarifications. Follow output instructions precisely."},
                {"role": "user", "content": interpretation_prompt}
            ],
            max_tokens=100, temperature=0.0
        ).choices[0].message.content.strip()
        thinking_log.append(f"LLM Interpretation of user's HITL response: '{llm_interpretation_response}'")

        if is_multitask_hitl:
            if llm_interpretation_response.lower() == "yes":
                thinking_log.append(f"Decision: User agreed to proceed with pending multi-task: '{pending_task_desc_before_handling}'")
                new_current_task_for_processing = pending_task_desc_before_handling
            else:
                thinking_log.append("Decision: User declined to proceed with pending multi-task.")
                next_final_response_if_declined = "Okay, I won't proceed with that. What else can I help you with today?"
        else: # Unclear query clarification
            if llm_interpretation_response != "NO_CLARIFICATION_PROVIDED":
                thinking_log.append(f"Decision: User provided clarification. New task query: '{llm_interpretation_response}'")
                new_current_task_for_processing = llm_interpretation_response
                 # If they clarified an unclear query, the "original_input" for this new attempt should be their clarification.
                original_overall_input = llm_interpretation_response # Update original_input to the clarified one
            else:
                thinking_log.append("Decision: User did not provide a useful clarification for the unclear query.")
                next_final_response_if_declined = "Okay, I understand. If you can provide more details or rephrase, I can try again. How else can I help?"
        
        return {
            "input": user_response_to_clarification, # The user's actual last message
            "original_input": original_overall_input, # Potentially updated if clarification was for unclear query
            "current_task_input": new_current_task_for_processing, # Null if declined, or the new task
            "final_response": next_final_response_if_declined, # Set only if user declines/HITL ends here
            "pending_tasks_description": None, # Cleared after handling
            "clarification_prompt_for_user": None,
            "is_awaiting_hitl_response": False, # This HITL cycle is done
            "hitl_resume_data": None,
            "thinking_log": thinking_log,
            "history": state.get("history", []) + ["handle_hitl_response_node"],
            "results": [], "mongo_query": None, "is_aggregation_result": None, # Reset search results
            "is_cheese_related": True if new_current_task_for_processing else state.get("is_cheese_related") # Re-affirm cheese if new task
        }

    # Routing functions (only mongo_search one is now complex enough to keep separate)
    def route_after_mongo_search(state: CheeseAgentState) -> Literal["pinecone_search", "check_hitl_completion"]:
        thinking_log = state.get("thinking_log", [])
        thinking_log.append("------------------- Routing Decision: After MongoDB Search -------------------")
        if (state.get("results") and len(state.get("results", [])) > 0) or state.get("is_aggregation_result"):
            decision = "check_hitl_completion"
            thinking_log.append(f"MongoDB found results or was aggregation. Routing to: {decision}")
            return decision
        else:
            decision = "pinecone_search"
            thinking_log.append("MongoDB found no results. Routing to: {decision}")
            return decision

    graph = StateGraph(CheeseAgentState)
    graph.add_node("check_cheese_related", check_cheese_related_node) # New entry point
    graph.add_node("preprocess_input", preprocess_input_and_detect_multitask_node)
    graph.add_node("convert_to_mongo_query", convert_to_mongo_query_node)
    graph.add_node("mongo_search", mongo_search_node)
    graph.add_node("pinecone_search", pinecone_search_node)
    graph.add_node("check_hitl_completion", check_completion_and_request_hitl_node)
    graph.add_node("handle_hitl_response", handle_hitl_response_node)
    graph.add_node("generate_response", generate_response_node)

    graph.set_entry_point("check_cheese_related")

    graph.add_conditional_edges(
        "check_cheese_related",
        lambda state: "preprocess_input" if state.get("is_cheese_related") or state.get("is_awaiting_hitl_response") else "generate_response",
        {
            "preprocess_input": "preprocess_input",
            "generate_response": "generate_response"
        }
    )

    graph.add_conditional_edges(
        "preprocess_input",
        lambda state: "handle_hitl_response" if state.get("hitl_resume_data") else "convert_to_mongo_query",
        {
            "handle_hitl_response": "handle_hitl_response",
            "convert_to_mongo_query": "convert_to_mongo_query"
        }
    )
    
    graph.add_conditional_edges(
        "convert_to_mongo_query",
        lambda state: END if state.get("is_awaiting_hitl_response") else "mongo_search", # Pause if unclear query sets this
        {
            "mongo_search": "mongo_search",
            END: END
        }
    )
    
    graph.add_conditional_edges(
        "mongo_search",
        route_after_mongo_search,
        {
            "check_hitl_completion": "check_hitl_completion",
            "pinecone_search": "pinecone_search"
        }
    )
    graph.add_edge("pinecone_search", "check_hitl_completion")

    graph.add_conditional_edges(
        "check_hitl_completion",
        lambda state: END if state.get("is_awaiting_hitl_response") else "generate_response", # Pause if multi-task sets this
        {
            "generate_response": "generate_response",
            END: END 
        }
    )

    graph.add_conditional_edges(
        "handle_hitl_response",
        lambda state: "convert_to_mongo_query" if state.get("current_task_input") else "generate_response", # If task, process; else (declined) final msg
        {
            "convert_to_mongo_query": "convert_to_mongo_query",
            "generate_response": "generate_response" 
        }
    )

    graph.add_edge("generate_response", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

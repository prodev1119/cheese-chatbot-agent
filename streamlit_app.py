import os
from PIL import Image
import streamlit as st
from agent.langgraph_agent import build_cheese_agent
from agent.mongo_search import MongoCheeseSearch
from agent.pinecone_search import PineconeCheeseSearch
from dotenv import load_dotenv
import uuid

load_dotenv()

# --- Set up your keys and configs here ---
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME") # Ensure this matches your index name on Pinecone
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Initialize searchers and agent ---
# Ensure API keys are loaded
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    st.error("Pinecone API Key or OpenAI API Key is missing. Please check your .env file or environment variables.")
    st.stop()

# Pass None for env if your Pinecone client version doesn't require it or handles it internally
mongo_search = MongoCheeseSearch(MONGO_URI, DB_NAME, COLLECTION_NAME)
pinecone_search = PineconeCheeseSearch(PINECONE_API_KEY, None, PINECONE_INDEX, OPENAI_API_KEY)
cheese_agent = build_cheese_agent(mongo_search, pinecone_search, OPENAI_API_KEY)

mermaid_code = cheese_agent.get_graph().draw_mermaid()
# print(mermaid_code) # You can use this output with a Mermaid renderer to create graph.png

# Set page config
st.set_page_config(
    page_title="Cheese Chatbot Agent",
    page_icon="🧀",
    # layout="wide" # Use wide layout to better accommodate sidebar
)

# --- Sidebar for Graph Display ---
with st.sidebar:
    st.header("Agent Workflow Graph")
    try:
        graph_image = Image.open("graph.png")
        st.image(graph_image, use_container_width=True,)
    except FileNotFoundError:
        st.warning("graph.png not found. Please generate it from the Mermaid code printed in the console.")
        # mermaid_code = cheese_agent.get_graph().draw_mermaid() # Regenerate if needed
        # st.text_area("Mermaid Code (for graph.png)", mermaid_code, height=300)

# --- Main Chat UI ---
st.title("🧀 Cheese Chatbot Agent")
st.write("Ask me anything about our cheese products!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    # Create a unique thread_id for the session, or load if you have persistence for it
    st.session_state.thread_id = str(uuid.uuid4())
    print(f"New session started with thread_id: {st.session_state.thread_id}")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display thinking log if it exists for this assistant message
        if message["role"] == "assistant" and "thinking_log" in message and message["thinking_log"]:
            with st.expander("Reasoning Process"):
                #st.text_area("Reasoning Steps", "\n".join(message["thinking_log"]), height=300, disabled=True)
                for line in message["thinking_log"]:
                    st.markdown(f"`{line}`") # Using markdown for better formatting, or st.text for plain

        # If products were associated with this assistant message, display them
        is_aggregation_history = message.get("is_aggregation_result", False)
        if not is_aggregation_history and "products" in message and message["products"]:
            with st.expander("View Details of All Products"):
                for cheese in message["products"]:
                    title = cheese.get("title", "No Title")
                    text = cheese.get("text", "No product description available.")
                    image_url = cheese.get("image_url")
                    product_url = cheese.get("product_url")
                    sku = cheese.get("sku")
                    PriceOrder = cheese.get("priceOrder")
                    PopularityOrder = cheese.get("popularityOrder")
                    Category = cheese.get("category")
                    Brand = cheese.get("brand")
                    PricePerLB = cheese.get("price_per_lb")
                    PricePerCT = cheese.get("price_per_ct")
                    EachPrice = cheese.get("each_price")
                    CasePrice = cheese.get("case_price")
                    EachSize = cheese.get("each_dimensions")
                    CaseSize = cheese.get("case_dimensions")
                    EachWeight = cheese.get("each_weight")
                    CaseWeight = cheese.get("case_weight")
                    EachCount = cheese.get("each_count")
                    CaseCount = cheese.get("case_count")

                    st.subheader(title)
                    if image_url:
                        st.image(image_url, width=150)
                    st.write(text)
                    st.write(f"SKU: {sku}")
                    st.write(f"Price Order: {PriceOrder}")
                    st.write(f"Popularity Order: {PopularityOrder}")
                    st.write(f"Category: {Category}")
                    st.write(f"Brand: {Brand}")
                    # st.write(f"Price Per LB: {PricePerLB}")
                    # st.write(f"Price Per CT: {PricePerCT}")
                    st.write(f"Each Price: {EachPrice}")
                    # st.write(f"Case Price: {CasePrice}")
                    # st.write(f"Each Size: {EachSize}")
                    # st.write(f"Case Size: {CaseSize}")
                    # st.write(f"Each Weight: {EachWeight}")
                    # st.write(f"Case Weight: {CaseWeight}")
                    # st.write(f"Each Count: {EachCount}")
                    # st.write(f"Case Count: {CaseCount}")
                    if product_url:
                        st.markdown(f"[Product Link]({product_url})")
                    st.markdown("---")

# Get user input
user_input = st.chat_input("Your question:")

if user_input:
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🧀 Thinking..."):
        # Define the initial state for the agent according to CheeseAgentState
        # Start with defaults
        agent_state_for_invoke = {
            "input": user_input, # This is the current text from the user
            "results": [],
            "is_cheese_related": None,
            "mongo_query": None,
            "final_response": None,
            "history": [], # History is managed by LangGraph checkpointer across invokes for the same thread_id
            "thinking_log": [], # New log for this specific invoke/resume
            "original_input": None,
            "current_task_input": None, 
            "pending_tasks_description": None,
            "clarification_prompt_for_user": None,
            "is_awaiting_hitl_response": False,
            "hitl_resume_data": None
        }

        # If the last message was an assistant message that set up HITL, carry over relevant state
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            last_assistant_message = st.session_state.messages[-1]
            if last_assistant_message.get("is_awaiting_hitl_response"):
                agent_state_for_invoke["is_awaiting_hitl_response"] = True
                agent_state_for_invoke["original_input"] = last_assistant_message.get("original_input")
                # current_task_input for resuming will be set by preprocess from user_input containing yes/no
                agent_state_for_invoke["pending_tasks_description"] = last_assistant_message.get("pending_tasks_description")
                agent_state_for_invoke["clarification_prompt_for_user"] = last_assistant_message.get("clarification_prompt_for_user")
                # `hitl_resume_data` will be set from `user_input` by `preprocess_input_and_detect_multitask_node`

        # Config for checkpointer (e.g., thread_id)
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        try:
            result_state = cheese_agent.invoke(agent_state_for_invoke, config=config)

            final_response_text = result_state.get("final_response", "Sorry, I couldn't process that.")
            products_to_display = result_state.get("results", [])
            is_aggregation_current = result_state.get("is_aggregation_result", False)
            current_thinking_log = result_state.get("thinking_log", [])
            # Check if agent is now awaiting HITL for the *next* turn
            now_awaiting_hitl = result_state.get("is_awaiting_hitl_response", False)
            clarification_q = result_state.get("clarification_prompt_for_user")

            # The final_response_text might be the clarification question itself if HITL is triggered
            if now_awaiting_hitl and clarification_q:
                # The agent has asked a clarification question and is paused.
                # final_response_text is already set to clarification_q by the HITL node.
                st.session_state.is_agent_paused = True # Custom flag for UI hints
            else:
                st.session_state.is_agent_paused = False

            assistant_message = {"role": "assistant", "content": final_response_text}
            if not is_aggregation_current and products_to_display and not now_awaiting_hitl:
                # Only show products if not an aggregation AND not currently a clarification question turn
                assistant_message["products"] = products_to_display
            
            assistant_message["is_aggregation_result"] = is_aggregation_current
            assistant_message["thinking_log"] = current_thinking_log
            # Persist key HITL state for next turn if agent paused
            if now_awaiting_hitl:
                assistant_message["is_awaiting_hitl_response"] = True
                assistant_message["clarification_prompt_for_user"] = clarification_q
                assistant_message["original_input"] = result_state.get("original_input")
                assistant_message["current_task_input"] = result_state.get("current_task_input")
                assistant_message["pending_tasks_description"] = result_state.get("pending_tasks_description")
            
            st.session_state.messages.append(assistant_message)

            with st.chat_message("assistant"):
                # Display thinking_log for the current response
                if current_thinking_log:
                    with st.expander("Reasoning Process"):
                        #st.text_area("Reasoning Steps", "\n".join(current_thinking_log), height=300, disabled=True)
                        for line in current_thinking_log:
                            st.markdown(f"`{line}`") # Using markdown for better formatting

                st.markdown(final_response_text)
                # Conditionally display products for the current response
                # Do not show "View Details" if it's a clarification question being asked
                if not is_aggregation_current and products_to_display and not now_awaiting_hitl:
                    with st.expander("View Details of All Products"):
                        for cheese in products_to_display:
                            title = cheese.get("title", "No Title")
                            text = cheese.get("text", "No product description available.")
                            image_url = cheese.get("image_url")
                            product_url = cheese.get("product_url")
                            sku = cheese.get("sku")
                            PriceOrder = cheese.get("priceOrder")
                            PopularityOrder = cheese.get("popularityOrder")
                            Category = cheese.get("category")
                            Brand = cheese.get("brand")
                            PricePerLB = cheese.get("price_per_lb")
                            PricePerCT = cheese.get("price_per_ct")
                            EachPrice = cheese.get("each_price")
                            CasePrice = cheese.get("case_price")
                            EachSize = cheese.get("each_dimensions")
                            CaseSize = cheese.get("case_dimensions")
                            EachWeight = cheese.get("each_weight")
                            CaseWeight = cheese.get("case_weight")
                            EachCount = cheese.get("each_count")
                            CaseCount = cheese.get("case_count")

                            st.subheader(title)
                            if image_url:
                                st.image(image_url, width=150)
                            st.write(text)
                            st.write(f"SKU: {sku}")
                            st.write(f"Price Order: {PriceOrder}")
                            st.write(f"Popularity Order: {PopularityOrder}")
                            st.write(f"Category: {Category}")
                            st.write(f"Brand: {Brand}")
                            # st.write(f"Price Per LB: {PricePerLB}")
                            # st.write(f"Price Per CT: {PricePerCT}")
                            st.write(f"Each Price: {EachPrice}")
                            # st.write(f"Case Price: {CasePrice}")
                            # st.write(f"Each Size: {EachSize}")
                            # st.write(f"Case Size: {CaseSize}")
                            # st.write(f"Each Weight: {EachWeight}")
                            # st.write(f"Case Weight: {CaseWeight}")
                            # st.write(f"Each Count: {EachCount}")
                            # st.write(f"Case Count: {CaseCount}")
                            if product_url:
                                st.markdown(f"[Product Link]({product_url})")
                            st.markdown("---")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})
            # No need to call st.rerun() explicitly, Streamlit handles chat_input updates

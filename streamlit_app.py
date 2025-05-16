import os
from PIL import Image
import streamlit as st
from agent.langgraph_agent import build_cheese_agent
from agent.mongo_search import MongoCheeseSearch
from agent.pinecone_search import PineconeCheeseSearch
from dotenv import load_dotenv

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
if not PINECONE_API_KEY or not OPENAI_API_KEY or not PINECONE_INDEX:
    missing_vars = []
    if not PINECONE_API_KEY:
        missing_vars.append("PINECONE_API_KEY")
    if not OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    if not PINECONE_INDEX:
        missing_vars.append("PINECONE_INDEX_NAME")
    st.error(f"The following environment variable(s) are missing: {', '.join(missing_vars)}. Please check your .env file or environment variables.")
    st.stop()

# Pass None for env if your Pinecone client version doesn't require it or handles it internally
mongo_search = MongoCheeseSearch(MONGO_URI, DB_NAME, COLLECTION_NAME)
pinecone_search = PineconeCheeseSearch(PINECONE_API_KEY, None, PINECONE_INDEX, OPENAI_API_KEY)
cheese_agent = build_cheese_agent(mongo_search, pinecone_search, OPENAI_API_KEY)

# mermaid_code = cheese_agent.get_graph().draw_mermaid()
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

# --- Main Chat UI ---
st.title("🧀 Cheese Chatbot Agent")
st.write("Ask me anything about our cheese products!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display thinking log if it exists for this assistant message
        if message["role"] == "assistant" and "thinking_log" in message and message["thinking_log"]:
            with st.expander("Reansoning Process"):
                #st.text_area("Reasoning Steps", "\n".join(message["thinking_log"]), height=300, disabled=True)
                for line in message["thinking_log"]:
                    st.markdown(f"`{line}`") # Using markdown for better formatting, or st.text for plain

        st.markdown(message["content"])

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
        initial_agent_state = {
            "input": user_input,
            "results": [],
            "is_cheese_related": None,
            "mongo_query": None,
            "final_response": None,
            "history": [],
            "thinking_log": [] # Initialize thinking_log
        }

        try:
            # Invoke the agent
            result_state = cheese_agent.invoke(initial_agent_state)

            final_response_text = result_state.get("final_response", "Sorry, I couldn't process that.")
            products_to_display = result_state.get("results", [])
            is_aggregation_current = result_state.get("is_aggregation_result", False)
            current_thinking_log = result_state.get("thinking_log", []) # Get the thinking log

            assistant_message = {"role": "assistant", "content": final_response_text}
            if not is_aggregation_current and products_to_display:
                assistant_message["products"] = products_to_display
            assistant_message["is_aggregation_result"] = is_aggregation_current
            assistant_message["thinking_log"] = current_thinking_log # Store thinking log
            st.session_state.messages.append(assistant_message)

            with st.chat_message("assistant"):
                # Display thinking_log for the current response
                if current_thinking_log:
                    with st.expander("Reansoning Process"):
                        #st.text_area("Reasoning Steps", "\n".join(current_thinking_log), height=300, disabled=True)
                        for line in current_thinking_log:
                            st.markdown(f"`{line}`") # Using markdown for better formatting

                st.markdown(final_response_text)
                # Conditionally display products for the current response
                if not is_aggregation_current and products_to_display:
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

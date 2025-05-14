import streamlit as st
from agent.langgraph_agent import build_cheese_agent
from agent.hybrid_search import HybridCheeseSearch
from agent.mongo_search import MongoCheeseSearch
from agent.pinecone_search import PineconeCheeseSearch

# --- Set up your keys and configs here ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "cheese_db"
COLLECTION_NAME = "cheeses"
PINECONE_API_KEY = "pcsk_22wZ3E_7iwmHQNLfKwwAbb6nM36KdQ69mJR3Dh6JdNVCPRLdPUPBT3wKKNdfkwrUHsDBog"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX = "cheese-products"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Initialize searchers and agent ---
mongo_search = MongoCheeseSearch(MONGO_URI, DB_NAME, COLLECTION_NAME)
pinecone_search = PineconeCheeseSearch(PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX, OPENAI_API_KEY)
hybrid_search = HybridCheeseSearch(mongo_search, pinecone_search)
cheese_agent = build_cheese_agent(hybrid_search)

# --- Streamlit UI ---
st.title("🧀 Cheese Chatbot Agent")
st.write("Ask me anything about our cheese products!")

user_input = st.text_input("Your question:")
if user_input:
    with st.spinner("Searching..."):
        # Only pass serializable data in the state!
        state = {"input": user_input}
        result_state = cheese_agent.invoke(state)
        results = result_state["results"]
        for cheese in results:
            st.subheader(cheese.get("title", "No Title"))
            st.write(cheese.get("text", ""))
            if "image_url" in cheese:
                st.image(cheese["image_url"], width=200)
            st.markdown(f"[Product Link]({cheese.get('product_url', '#')})")
            st.markdown("---")

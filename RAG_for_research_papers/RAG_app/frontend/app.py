import streamlit as st
import requests
import json
import os

# --- Configuration ---
# FastAPI service address
# For local development: localhost
# For Docker: use service name "backend"
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "localhost")
FASTAPI_PORT = os.getenv("FASTAPI_PORT", "8000")
API_ENDPOINT = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/query"

st.write(f"🔗 Connecting to backend at: {API_ENDPOINT}")

# --- Streamlit UI ---
st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")

st.title("🤖 Gemini RAG Chatbot")
st.markdown("Ask a question and get an answer grounded in the knowledge base (via FastAPI & ChromaDB).")

# Input field for user query
user_query = st.text_input("Enter your query:", key="query_input")

# Function to handle the API request
def fetch_rag_answer(query):
    try:
        # Send POST request to the FastAPI backend
        response = requests.post(
            API_ENDPOINT,
            json={"user_query": query},
            timeout=30  # Increased timeout for LLM generation
        )
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the backend service at {API_ENDPOINT}. Ensure the Docker containers are running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

if st.button("Get RAG Answer"):
    if not user_query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating RAG answer..."):
            result = fetch_rag_answer(user_query)
            
            if result:
                # Display the answer
                st.subheader("Answer")
                st.info(result.get("answer", "No answer received."))
                
                # Display the contexts
                st.subheader("Contexts Used")
                contexts = result.get("contexts", [])
                if contexts:
                    for i, context in enumerate(contexts):
                        st.markdown(f"**Chunk {i+1}:** {context}")
                else:
                    st.markdown("_No contexts were retrieved or used._")

# Optional: Display API endpoint status
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Backend Endpoint:** `{API_ENDPOINT}`")
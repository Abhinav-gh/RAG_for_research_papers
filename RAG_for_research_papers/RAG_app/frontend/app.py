import streamlit as st
import requests
import json
import os

# --- Configuration ---
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "localhost")
FASTAPI_PORT = os.getenv("FASTAPI_PORT", "8001")
API_ENDPOINT = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/query"

st.write(f"🔗 Connecting to backend at: {API_ENDPOINT}")

# --- Streamlit UI ---
st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")

st.title("🤖 Gemini RAG Chatbot")
st.markdown("Ask a question and get an answer grounded in the knowledge base.")

# Query input
user_query = st.text_input("Enter your question:", key="query_input")

# Optional settings (use backend defaults if you want)
top_k = st.number_input("Top-K chunks to retrieve", min_value=1, max_value=50, value=10)
top_m = st.number_input("Top-M chunks after reranking", min_value=1, max_value=10, value=3)

# Function to call backend
def fetch_rag_answer(query, top_k, top_m):
    payload = {
        "query": query,
        "top_k": int(top_k),
        "top_m": int(top_m)
    }

    try:
        response = requests.post(API_ENDPOINT, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to backend at {API_ENDPOINT}. Is it running?")
        return None

    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.status_code} - {e.response.text}")
        return None

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# Button action
if st.button("Get RAG Answer"):
    if not user_query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Working..."):
            result = fetch_rag_answer(user_query, top_k, top_m)

            if result:
                # Show answer
                st.subheader("📘 Answer")
                st.info(result.get("answer", "No answer provided."))

                # Show contexts
                st.subheader("📚 Contexts Used")
                contexts = result.get("contexts", [])
                if contexts:
                    for idx, context in enumerate(contexts):
                        st.markdown(f"**Context {idx+1}:**\n{context}\n---")
                else:
                    st.markdown("_No contexts returned by the backend._")


# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Backend:** `{API_ENDPOINT}`")

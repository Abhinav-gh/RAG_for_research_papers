import streamlit as st
import requests
import os

# Backend endpoints
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "localhost")
FASTAPI_PORT = os.getenv("FASTAPI_PORT", "8001")
BASE_URL = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"

QUERY_ENDPOINT = f"{BASE_URL}/query"
HEALTH_ENDPOINT = f"{BASE_URL}/health"
MODEL_INFO_ENDPOINT = f"{BASE_URL}/model_info"

st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")

st.title("🤖 Gemini RAG Dashboard")

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def api_get(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ GET error: {e}")
        return None

def api_post(url, payload):
    try:
        r = requests.post(url, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ POST error: {e}")
        return None


# ---------------------------------------------------------
# Create tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["💬 RAG Query", "🔍 Backend Health", "🧠 Model Info"]
)

# ---------------------------------------------------------
# TAB 1 — RAG QUERY
# ---------------------------------------------------------
with tab1:
    st.subheader("💬 Ask a Question")

    user_query = st.text_input("Enter your question:", key="query_input")

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.number_input("Top-K retrieve", min_value=1, max_value=50, value=10)
    with col2:
        top_m = st.number_input("Top-M rerank", min_value=1, max_value=10, value=3)

    if st.button("Run RAG", key="btn_rag_run"):
        if not user_query:
            st.warning("Enter a question first.")
        else:
            with st.spinner("Processing RAG query..."):
                payload = {"query": user_query, "top_k": int(top_k), "top_m": int(top_m)}
                result = api_post(QUERY_ENDPOINT, payload)

                if result:
                    st.markdown("### 📘 Answer & Contexts")

                    answer = result.get("answer", "").strip()

                    st.markdown("### 📘 Answer")

                    if not answer:
                        st.warning("⚠ No answer generated.")
                    else:
                        # Normalize whitespace & preserve line breaks
                        formatted_answer = answer.replace("\n", "<br>")

                        st.markdown(
                            f"""
                            <div style="
                                background-color:#1e3d2f;
                                border-left:4px solid #4caf50;
                                padding:15px;
                                color:#e8f5e9;
                                border-radius:8px;
                                margin-bottom:20px;
                                font-size:16px;
                                line-height:1.55;
                                white-space:normal;
                            ">
                            {formatted_answer}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    # Contexts
                    contexts = result.get("contexts", [])
                    if not contexts:
                        st.info("No contexts returned.")
                    else:
                        for idx, ctx in enumerate(contexts):
                            st.markdown(
                                f"""
                                <div style="
                                    background-color:#2b2b2b;
                                    border-left:4px solid #7e57c2;
                                    padding:14px;
                                    border-radius:6px;
                                    color:#e0e0e0;
                                    margin-bottom:14px;
                                    font-size:15px;
                                    line-height:1.45;
                                    white-space:pre-wrap;
                                    overflow-wrap:break-word;
                                    word-break:break-word;
                                    max-height:none;
                                ">
                                    <strong style="color:#b39ddb;">Context {idx+1}</strong><br><br>
                                    {ctx}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )


# ---------------------------------------------------------
# TAB 2 — HEALTH
# ---------------------------------------------------------
with tab2:
    st.subheader("🔍 Backend Health Check")

    if st.button("Run Health Check"):
        resp = api_get(HEALTH_ENDPOINT)
        if resp:
            st.success("Backend is healthy!")
            st.json(resp)

# ---------------------------------------------------------
# TAB 3 — MODEL INFO
# ---------------------------------------------------------
with tab3:
    st.subheader("🧠 Loaded Model Information")

    if st.button("Show Model Info"):
        resp = api_get(MODEL_INFO_ENDPOINT)
        if resp:
            st.json(resp)

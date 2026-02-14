import streamlit as st
import os
import time

from rag.loader import load_pdf
from rag.indexer import build_index
from rag.retriever import retrieve_context
from rag.generator import load_model, generate_response
from rag.prompts import build_prompt
from utils.metrics import approximate_token_count


# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="PDF Conversational RAG", layout="wide")
st.title("📄 PDF Conversational Intelligence (Llama 3 RAG)")


# ------------------------------
# Session State Initialization
# ------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "index" not in st.session_state:
    st.session_state.index = None

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None


# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("⚙️ Retrieval Settings")

chunk_size = st.sidebar.slider("Chunk Size", 300, 1200, 800, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 300, 100, step=50)
top_k = st.sidebar.slider("Top K Chunks", 1, 5, 3)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, step=0.1)

show_context = st.sidebar.checkbox("Show Retrieved Context")


# ------------------------------
# Load Llama Model (Cached)
# ------------------------------
tokenizer, model = load_model()


# ------------------------------
# PDF Upload Section
# ------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # If new PDF uploaded → reset session
    if uploaded_file.name != st.session_state.current_pdf:

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Processing and indexing document..."):
            documents = load_pdf("temp.pdf")

            st.session_state.index = build_index(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        st.session_state.chat_history = []
        st.session_state.question_count = 0
        st.session_state.current_pdf = uploaded_file.name

        st.success("Document indexed successfully!")


# ------------------------------
# Display Existing Chat History
# ------------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ------------------------------
# Chat Input
# ------------------------------
if prompt := st.chat_input("Ask a question about the document..."):

    # Guard: No document
    if st.session_state.index is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    # Enforce 10-question limit
    if st.session_state.question_count >= 10:
        st.warning("Maximum of 10 questions reached. Resetting session.")
        st.session_state.chat_history = []
        st.session_state.question_count = 0
        st.stop()

    st.session_state.question_count += 1

    # Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append(
        {"role": "user", "content": prompt}
    )

    # ------------------------------
    # Retrieval
    # ------------------------------
    retrieved_chunks = retrieve_context(
        st.session_state.index,
        prompt,
        top_k=top_k
    )

    document_context = "\n\n".join(
        [chunk["text"] for chunk in retrieved_chunks]
    )

    # Keep only last 2 exchanges (4 messages total)
    recent_history = st.session_state.chat_history[-4:]

    conversation_context = ""
    for msg in recent_history:
        conversation_context += f"{msg['role']}: {msg['content']}\n"

    # ------------------------------
    # Prompt Construction
    # ------------------------------
    final_prompt = build_prompt(
        conversation_context,
        document_context,
        prompt
    )

    # ------------------------------
    # Generation
    # ------------------------------
    response, latency = generate_response(
        tokenizer,
        model,
        final_prompt,
        temperature=temperature
    )

    # Clean response extraction
    answer = response.replace(final_prompt, "").strip()

    token_estimate = approximate_token_count(answer)

    # ------------------------------
    # Display Assistant Message
    # ------------------------------
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(
            f"⏱ Response time: {latency} seconds | 🔢 Approx tokens: {token_estimate}"
        )

        if show_context:
            with st.expander("Retrieved Context"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(
                        f"**Chunk {i+1} (Score: {chunk['score']:.4f})**"
                    )
                    st.write(chunk["text"])

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

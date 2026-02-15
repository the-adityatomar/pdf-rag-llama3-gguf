import os
import time
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from rag.loader import load_pdf
from rag.indexer import build_index
from rag.retriever import retrieve_context
from rag.prompts import build_prompt
from utils.metrics import approximate_token_count


# ----------------------------
# Download GGUF Model
# ----------------------------
MODEL_REPO = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
MODEL_FILENAME = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

print("Downloading GGUF model (first run may take time)...")

MODEL_PATH = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILENAME
)

print("Model downloaded.")


# ----------------------------
# Load Llama.cpp Model (CPU)
# ----------------------------
print("Loading model into memory...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=os.cpu_count(),
    n_gpu_layers=0  # CPU only
)

print("Model loaded successfully.")


# ----------------------------
# Global State
# ----------------------------
index = None
chat_history = []
question_count = 0
current_pdf = None


# ----------------------------
# PDF Processing
# ----------------------------
def process_pdf(file, chunk_size, chunk_overlap):
    global index, chat_history, question_count, current_pdf

    if file is None:
        return "Please upload a PDF."

    if file.name != current_pdf:

        documents = load_pdf(file.name)

        index = build_index(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chat_history = []
        question_count = 0
        current_pdf = file.name

    return "Document indexed successfully."


# ----------------------------
# Chat Function
# ----------------------------
def chat_function(message, history, chunk_size, chunk_overlap, top_k, temperature):
    global index, chat_history, question_count

    if index is None:
        return "Upload a PDF first."

    if question_count >= 10:
        chat_history = []
        question_count = 0
        return "Maximum 10 questions reached. Session reset."

    question_count += 1

    start_time = time.time()

    # Retrieval
    retrieved_chunks = retrieve_context(index, message, top_k=top_k)

    document_context = "\n\n".join(
        [chunk["text"] for chunk in retrieved_chunks]
    )

    # Keep only last 2 exchanges
    recent_history = chat_history[-4:]

    conversation_context = ""
    for msg in recent_history:
        conversation_context += f"{msg['role']}: {msg['content']}\n"

    # Build Prompt
    final_prompt = build_prompt(
        conversation_context,
        document_context,
        message
    )

    # Generate
    output = llm(
        final_prompt,
        max_tokens=512,
        temperature=temperature,
        top_p=0.9
    )

    answer = output["choices"][0]["text"].strip()

    latency = round(time.time() - start_time, 2)
    token_estimate = approximate_token_count(answer)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})

    return f"{answer}\n\n⏱ {latency}s | 🔢 ~{token_estimate} tokens"


# ----------------------------
# Gradio Interface
# ----------------------------
with gr.Blocks() as demo:

    gr.Markdown("# 📄 PDF Conversational Intelligence (Llama 3 GGUF - CPU)")

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])

    with gr.Row():
        chunk_size = gr.Slider(300, 1200, value=800, step=100, label="Chunk Size")
        chunk_overlap = gr.Slider(0, 300, value=100, step=50, label="Chunk Overlap")
        top_k = gr.Slider(1, 5, value=3, step=1, label="Top K")
        temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="Temperature")

    process_button = gr.Button("Process PDF")
    status_output = gr.Textbox(label="Status")

    process_button.click(
        process_pdf,
        inputs=[pdf_file, chunk_size, chunk_overlap],
        outputs=status_output
    )

    chatbot = gr.ChatInterface(
        fn=lambda message, history: chat_function(
            message,
            history,
            chunk_size.value,
            chunk_overlap.value,
            top_k.value,
            temperature.value
        )
    )

demo.launch()



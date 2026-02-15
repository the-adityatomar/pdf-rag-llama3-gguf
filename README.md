---
title: PDF Conversational RAG (Llama 3 GGUF)
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.36.1"
app_file: app.py
pinned: false
---
# 📄 PDF Conversational RAG (Llama 3 GGUF – CPU Optimized)

## 🚀 Overview

This project implements a **Conversational Retrieval-Augmented Generation (RAG) system** that allows users to upload a PDF and interact with it through a chat interface.

The system is optimized for **CPU deployment** using:

- Quantized Llama 3 (GGUF format)
- `llama.cpp` inference backend
- FAISS vector retrieval
- MiniLM embeddings
- Gradio interface

It is designed to run efficiently on **Hugging Face CPU Spaces**, without requiring GPU acceleration.

---

## 🧠 How It Works

1. User uploads a PDF  
2. The document is chunked into smaller segments  
3. Each chunk is embedded using MiniLM  
4. FAISS builds an in-memory vector index  
5. User asks a question  
6. Top-K relevant chunks are retrieved  
7. A structured prompt is constructed  
8. Llama 3 GGUF generates a grounded answer  

The model is instructed to answer **only from the document context**.

---

## ⚙️ Core Features

- 📄 Upload any PDF
- 💬 Conversational chat interface
- 🔁 Maximum 10 consecutive questions per session
- 🧠 Only the last 2 exchanges are retained for context
- 🎛 Adjustable:
  - Chunk size
  - Chunk overlap
  - Top-K retrieval
  - Temperature
- 📊 Response latency reporting
- 🔢 Approximate token count display
- 💻 Fully CPU-compatible deployment

---

## 🏗 System Architecture


---

## 🧩 Tech Stack

### Language Model
- Meta Llama 3 8B (GGUF Q4_K_M)
- Backend: `llama.cpp` via `llama-cpp-python`

### Retrieval
- sentence-transformers/all-MiniLM-L6-v2
- FAISS (in-memory vector search)

### Framework
- LlamaIndex (document loading + indexing pipeline)

### Interface
- Gradio

### Deployment Target
- Hugging Face Spaces (CPU)

---

## 📦 Installation (Local Development)

Install dependencies:

```bash
pip install -r requirements.txt
```
---

You can now:

```bash
git add README.md
git commit -m "Updated README for GGUF CPU architecture"
git push
```

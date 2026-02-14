# PDF Conversational Intelligence System (Llama 3 RAG)

## Overview
A GPU-powered conversational Retrieval-Augmented Generation (RAG) system that allows users to upload any PDF and interact with it using Meta Llama 3 8B.

## Architecture

User → Streamlit UI  
PDF → Chunking → MiniLM Embeddings → FAISS  
Query → Retrieval → Prompt Assembly  
Prompt → Llama 3 (4-bit quantized) → Response  

## Core Features

- Conversational chat interface
- Maximum 10 consecutive questions per session
- Only last 2 exchanges retained for context
- Adjustable chunk size, overlap, top-k, temperature
- Retrieved chunk transparency
- Latency reporting
- Approximate token estimation
- Session-based document indexing

## Tech Stack

- LLM: Meta-Llama-3-8B-Instruct
- Embeddings: all-MiniLM-L6-v2
- Vector Store: FAISS (in-memory)
- Framework: LlamaIndex
- UI: Streamlit
- Deployment: Hugging Face Spaces (GPU required)

## Deployment Steps

1. Push repo to GitHub
2. Create Hugging Face Space
3. Select:
   - SDK: Streamlit
   - Hardware: GPU (T4)
4. Add Hugging Face token in Secrets
5. Deploy

## System Constraints

- Single document per session
- Max 10 questions per upload
- Retrieval based only on current query
- Conversation memory limited to last 2 exchanges
# PDF Conversational Intelligence System (Llama 3 RAG)

## Overview
A GPU-powered conversational RAG system that allows users to upload a PDF and interact with it using Llama 3.

## Architecture
- LLM: Meta Llama 3 8B Instruct
- Embeddings: MiniLM
- Vector Store: FAISS (in-memory)
- Framework: LlamaIndex
- UI: Streamlit
- Deployment: Hugging Face Spaces (GPU)

## Features
- Chat-style interface
- Up to 10 consecutive questions
- Context from last 2 exchanges
- Adjustable retrieval settings
- Retrieved chunk transparency
- Latency reporting

## Setup
1. Clone repo
2. Install requirements
3. Add HF token to `.env`
4. Run: `streamlit run app.py`

## Deployment
Deploy via Hugging Face Spaces (Streamlit SDK, GPU enabled).

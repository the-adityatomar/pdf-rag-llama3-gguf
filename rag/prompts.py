def build_prompt(conversation_context, document_context, user_query):
    """
    Constructs final prompt injected into Llama.
    """

    return f"""
You are a document-based assistant.

STRICT RULES:
1. Use ONLY the provided document context.
2. If the answer is not present, say:
   "The document does not contain this information."
3. Do not use outside knowledge.

Conversation History:
{conversation_context}

Document Context:
{document_context}

User Question:
{user_query}

Answer:
"""

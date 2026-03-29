"""
Configuration settings for the RAG application.
All configurable parameters are centralized here.
"""

import os
from pathlib import Path


BASE_DIR = Path(__file__).parents

# Document paths
DOCUMENTS_DIR = BASE_DIR / "documents"
DEFAULT_DOCUMENT_PATH = DOCUMENTS_DIR / "document.pdf"

# ChromaDB settings
CHROMA_DB_PERSIST_DIR = BASE_DIR / "chroma_db"

# Text chunking settings
CHUNK_SIZE = 800  # characters (500-1000 range)
CHUNK_OVERLAP = 100  # characters

# Retrieval settings
TOP_K_RETRIEVAL = 5  # Number of chunks to retrieve

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# LLM Settings
LLM_MODEL_NAME = "openai/gpt-oss-120b"  # HuggingFace Inference API model
LLM_CONTEXT_LENGTH = 4096
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 512
HF_API_KEY = "hf_xxxxxxxxxxxxx"  # Your HuggingFace API key

# Response control
TONE_OPTIONS = ["Simple", "Formal"]
LENGTH_OPTIONS = ["Short", "Detailed"]

# Document priority levels
PRIORITY_ORDER = ["Bible", "Notes", "General"]
PRIORITY_MAP = {
    "Bible": "high",
    "Notes": "medium",
    "General": "low"
}

# System prompt template
STRICT_SYSTEM_PROMPT = """You are a precise assistant that answers ONLY from the provided context.

RULES:
1. Use ONLY information explicitly stated in the context
2. If the answer is not in context, respond: "I don't know based on the provided documents"
3. Do not use prior knowledge or external information
4. Do not speculate or infer beyond what's stated
5. Cite specific sections if available

Context:
{context}

User Question: {question}

Response Settings:
- Tone: {tone}
- Length: {length}
- Include references: {include_references}

Answer:"""

# Tone guidelines
TONE_GUIDLINES = {
    "Simple": "Use casual, easy-to-understand language. Keep it conversational.",
    "Formal": "Use professional, formal language with complete sentences and proper terminology."
}

# Length guidelines
LENGTH_GUIDELINES = {
    "Short": "Provide a concise answer in 2-3 sentences. Focus on key points only.",
    "Detailed": "Provide a comprehensive answer in 4-6 sentences. Include examples, explanations, and elaboration where relevant."
}

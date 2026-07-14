"""
RAG sub-package for Smart MCQ Solver.

Modules
-------
vector_store   : FAISS / Chroma vector database wrapper
retriever      : Semantic retrieval logic
prompt_builder : Prompt template construction
rag_scorer     : LLM-based answer scoring via RAG
pipeline       : End-to-end RAG pipeline orchestration
"""

from src.rag.vector_store import VectorStore
from src.rag.retriever import SemanticRetriever
from src.rag.prompt_builder import PromptBuilder
from src.rag.rag_scorer import RAGScorer
from src.rag.pipeline import RAGPipeline

__all__ = [
    "VectorStore",
    "SemanticRetriever",
    "PromptBuilder",
    "RAGScorer",
    "RAGPipeline",
]
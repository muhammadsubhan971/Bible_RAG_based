"""
Main RAG pipeline module that orchestrates document processing, retrieval, and generation.
Provides a unified interface for querying documents with strict context-based responses.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import config
from document_processor import process_document, get_all_pdf_files
from vector_store import initialize_chroma, VectorStore
from llm_handler import LLMHandler


class RAGPipeline:
    """Complete RAG pipeline for document-based question answering."""
    
    def __init__(self):
        """Initialize the RAG pipeline with all components."""
        print("=" * 60)
        print("Initializing RAG Pipeline...")
        print("=" * 60)
        
        # Initialize vector store
        self.vector_store = initialize_chroma()
        
        # Initialize LLM
        self.llm_handler = LLMHandler()
        
        # Track loaded documents
        self.loaded_documents = []

        print("\n✓ RAG Pipeline initialized successfully")
        print("=" * 60)
    
    def add_document(self, file_path: str, priority: str = "General", 
                    doc_type: str = None) -> int:
        """
        Add a document to the knowledge base.
        
        Args:
            file_path: Path to the PDF file
            priority: Document priority ("Bible", "Notes", "General")
            doc_type: Document type (defaults to priority value)
            
        Returns:
            Number of chunks added
        """
        print(f"\nAdding document: {file_path}")
        print(f"Priority: {priority}")
        
        try:
            # Process the document
            documents = process_document(file_path, priority=priority, doc_type=doc_type)
            
            if not documents:
                print("⚠ No text extracted from document")
                return 0
            
            # Add to vector store
            self.vector_store.add_documents(documents, priority=priority)
            
            # Track loaded document
            self.loaded_documents.append({
                "path": file_path,
                "chunks": len(documents),
                "priority": priority
            })
            
            print(f"✓ Successfully added {len(documents)} chunks from {Path(file_path).name}")
            return len(documents)
            
        except Exception as e:
            print(f"✗ Error adding document: {e}")
            return 0
    
    def index_all_documents(self, directory: str = None) -> int:
        """
        Index all PDF files in a directory.
        
        Args:
            directory: Directory to scan (defaults to DOCUMENTS_DIR)
            
        Returns:
            Total number of chunks indexed
        """
        if directory is None:
            directory = str(config.DOCUMENTS_DIR)
        
        print(f"\nIndexing all PDFs in: {directory}")
        
        # Get all PDF files
        pdf_files = get_all_pdf_files(directory)
        
        if not pdf_files:
            print("⚠ No PDF files found")
            return 0
        
        total_chunks = 0
        for pdf_path in pdf_files:
            # Determine priority based on filename
            filename_lower = pdf_path.name.lower()
            if "bible" in filename_lower:
                priority = "Bible"
            elif "note" in filename_lower:
                priority = "Notes"
            else:
                priority = "General"
            
            chunks = self.add_document(str(pdf_path), priority=priority)
            total_chunks += chunks
        
        print(f"\n✓ Indexed {len(pdf_files)} files, {total_chunks} total chunks")
        return total_chunks
    
    def query(self, question: str, tone: str = "Simple", 
              length: str = "Short", include_references: bool = False,
              priority_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            tone: Response tone ("Simple" or "Formal")
            length: Response length ("Short" or "Detailed")
            include_references: Whether to include references
            priority_filter: Optional filter by document type
            
        Returns:
            Dictionary with answer and metadata
        """
        print(f"\nQuery: {question}")
        print(f"Tone: {tone}, Length: {length}")
        
        # Retrieve relevant chunks
        chunks = self.vector_store.similarity_search(
            query=question,
            k=config.TOP_K_RETRIEVAL,
            priority_filter=priority_filter
        )
        
        if not chunks:
            return {
                "answer": "I don't have enough information in my knowledge base to answer this question.",
                "sources": [],
                "retrieved_chunks": 0
            }
        
        # Build context from retrieved chunks
        context_parts = []
        sources = set()
        
        for idx, chunk in enumerate(chunks):
            context_parts.append(f"[Chunk {idx+1}]\n{chunk.page_content}")
            sources.add(chunk.metadata.get("source", "Unknown"))
            
            # Add reference info if requested
            if include_references:
                context_parts[-1] += f"\n(Ref: {chunk.metadata.get('source', 'Unknown')}, Chunk {chunk.metadata.get('chunk_index', '?')})"
        
        context = "\n\n".join(context_parts)
        
        # Build the strict prompt
        prompt = self._build_prompt(
            question=question,
            context=context,
            tone=tone,
            length=length,
            include_references=include_references
        )
        
        # Generate answer using LLM
        answer = self.llm_handler.generate(
            prompt=prompt,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE
        )
        
        # Prepare response
        response = {
            "answer": answer,
            "sources": sorted(list(sources)),
            "retrieved_chunks": len(chunks),
            "context_used": context if len(context) < 5000 else context[:5000] + "..."
        }
        
        print(f"✓ Answer generated from {len(chunks)} chunks")
        print(f"Sources: {response['sources']}")
        
        return response
    
    def _build_prompt(self, question: str, context: str, tone: str, 
                     length: str, include_references: bool) -> str:
        """
        Build the complete prompt with context and instructions.
        
        Args:
            question: User question
            context: Retrieved context chunks
            tone: Response tone
            length: Response length
            include_references: Include references flag
            
        Returns:
            Complete formatted prompt
        """
        # Get tone and length guidelines
        tone_guideline = config.TONE_GUIDLINES.get(tone, "")
        length_guideline = config.LENGTH_GUIDELINES.get(length, "")
        
        # Build the prompt
        prompt = f"""You are a precise assistant that answers ONLY from the provided context.

RULES:
1. Use ONLY information explicitly stated in the context
2. If the answer is not in context, respond: "I don't know based on the provided documents"
3. Do not use prior knowledge or external information
4. Do not speculate or infer beyond what's stated
5. Cite specific sections or chunk numbers if available

Context:
{context}

User Question: {question}

Response Settings:
- Tone: {tone} - {tone_guideline}
- Length: {length} - {length_guideline}
- Include references: {include_references}

Answer:"""
        
        return prompt
    
    def clear_knowledge_base(self):
        """Clear all documents from the vector store."""
        self.vector_store.clear_database()
        self.loaded_documents = []
        print("✓ Knowledge base cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current knowledge base."""
        return {
            "total_chunks": self.vector_store.get_document_count(),
            "loaded_documents": len(self.loaded_documents),
            "documents": self.loaded_documents,
            "sources": self.vector_store.get_all_sources()
        }


def create_pipeline() -> RAGPipeline:
    """Create and return a new RAG pipeline instance."""
    return RAGPipeline()


if __name__ == "__main__":
    # Test the RAG pipeline
    print("Testing RAG Pipeline...\n")
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Check if default document exists
    if config.DEFAULT_DOCUMENT_PATH.exists():
        print(f"Found default document: {config.DEFAULT_DOCUMENT_PATH}")
        
        # Add the document
        chunks = pipeline.add_document(str(config.DEFAULT_DOCUMENT_PATH), priority="General")
        print(f"Added {chunks} chunks")
        
        # Test query
        print("\n" + "=" * 60)
        print("Test Query:")
        print("=" * 60)
        
        test_question = "What is the main topic of this document?"
        result = pipeline.query(
            question=test_question,
            tone="Simple",
            length="Short"
        )
        
        print(f"\nQuestion: {test_question}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
    else:
        print(f"Default document not found: {config.DEFAULT_DOCUMENT_PATH}")
        print("Add documents to the 'documents' folder and run again.")
    
    # Show stats
    stats = pipeline.get_stats()
    print(f"\nKnowledge Base Stats: {stats}")

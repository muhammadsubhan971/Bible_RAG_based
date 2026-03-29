"""
Vector store module using ChromaDB for semantic search.
Handles document storage, embeddings, and retrieval with priority filtering.
"""

from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import config


class VectorStore:
    """ChromaDB vector store wrapper with priority-based retrieval."""
    
    def __init__(self, persist_dir: str = None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_dir: Directory for persistent storage (default: from config)
        """
        if persist_dir is None:
            persist_dir = str(config.CHROMA_DB_PERSIST_DIR)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Initialize embedding model
        self._initialize_embeddings()
        
        print(f"✓ Vector store initialized at {persist_dir}")
    
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        print("Loading embedding model...")
        
        # Use environment variable for HF_TOKEN if available
        import os
        hf_token = os.getenv('HF_TOKEN')
        
        if hf_token:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                token=hf_token  # Use HF token for faster downloads
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        print("✓ Embedding model loaded")

    def add_documents(self, documents: List[Document], priority: str = "General"):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            priority: Priority level for all documents
        """
        if not documents:
            print("⚠ No documents to add")
            return 0
        
        # Process in smaller batches to avoid memory issues
        batch_size = 50  # Reduced batch size for stability
        total_docs = len(documents)
        added_count = 0
        
        print(f"Adding {total_docs} document chunks in batches of {batch_size}...")
        
        try:
            for i in range(0, total_docs, batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_docs + batch_size - 1) // batch_size
                
                print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_docs)} chunks)...")
                
                # Prepare data for ChromaDB
                ids = []
                texts = []
                metadatas = []
                
                for idx, doc in enumerate(batch_docs):
                    # Create unique ID
                    doc_id = f"{doc.metadata.get('source', 'doc')}_{doc.metadata.get('chunk_index', idx)}"
                    
                    ids.append(doc_id)
                    texts.append(doc.page_content)
                    
                    # Build metadata dict
                    metadata = {
                        "source": doc.metadata.get("source", "unknown"),
                        "chunk_index": doc.metadata.get("chunk_index", idx),
                        "priority": config.PRIORITY_MAP.get(priority, "low"),
                        "doc_type": doc.metadata.get("doc_type", priority),
                        "full_path": doc.metadata.get("full_path", ""),
                        "timestamp": doc.metadata.get("timestamp", "")
                    }
                    metadatas.append(metadata)
                
                # Generate embeddings
                embedded_texts = self.embeddings.embed_documents(texts)
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embedded_texts,
                    documents=texts,
                    metadatas=metadatas
                )
                
                added_count += len(batch_docs)
                print(f"    ✓ Batch {batch_num} complete")
            
            print(f"✓ Added {added_count} document chunks to vector store")
            return added_count
            
        except Exception as e:
            print(f"✗ Error adding documents: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def similarity_search(self, query: str, k: int = None, 
                         priority_filter: Optional[str] = None) -> List[Document]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query text
            k: Number of results to retrieve (default: from config)
            priority_filter: Optional filter by document type ("Bible", "Notes", "General")
            
        Returns:
            List of relevant Document objects
        """
        if k is None:
            k = config.TOP_K_RETRIEVAL
        
        # Generate embedding for query
        query_embedding = self.embeddings.embed_query(query)
        
        # Build where clause for priority filtering
        where_clause = None
        if priority_filter:
            where_clause = {"doc_type": priority_filter}
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2,  # Get more results initially for filtering
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to Document objects
        documents = []
        if results['documents'] and results['documents'][0]:
            for idx, (text, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                doc = Document(
                    page_content=text,
                    metadata={
                        **metadata,
                        "relevance_score": 1 - distance,  # Convert distance to similarity
                        "rank": idx + 1
                    }
                )
                documents.append(doc)
        
        # Apply priority-based ranking if no filter was applied
        if priority_filter is None and documents:
            documents = self._rank_by_priority(documents)
        
        # Return top-k after ranking
        return documents[:k]
    
    def _rank_by_priority(self, documents: List[Document]) -> List[Document]:
        """
        Re-rank documents based on priority and relevance.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Re-ranked list of Document objects
        """
        # Priority weights
        priority_weights = {
            "high": 1.3,    # Bible
            "medium": 1.1,  # Notes
            "low": 1.0      # General
        }
        
        # Score each document
        scored_docs = []
        for doc in documents:
            base_score = doc.metadata.get("relevance_score", 0.5)
            priority = doc.metadata.get("priority", "low")
            weight = priority_weights.get(priority, 1.0)
            
            # Adjusted score = relevance * priority_weight
            adjusted_score = base_score * weight
            doc.metadata["adjusted_score"] = adjusted_score
            scored_docs.append(doc)
        
        # Sort by adjusted score (descending)
        scored_docs.sort(key=lambda x: x.metadata.get("adjusted_score", 0), reverse=True)
        
        return scored_docs
    
    def clear_database(self):
        """Delete all documents from the vector store."""
        try:
            # Delete the collection
            self.client.delete_collection(name="rag_documents")
            # Recreate empty collection
            self.collection = self.client.create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"}
            )
            print("✓ Vector store cleared")
        except Exception as e:
            print(f"⚠ Error clearing database: {e}")
    
    def get_document_count(self) -> int:
        """Get total number of documents in the store."""
        try:
            count = self.collection.count()
            return count
        except Exception:
            return 0
    
    def get_all_sources(self) -> List[str]:
        """Get list of all unique document sources."""
        try:
            # Get a sample to extract sources
            results = self.collection.get(
                include=["metadatas"],
                limit=1000
            )
            
            sources = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    sources.add(metadata.get('source', 'unknown'))
            
            return sorted(list(sources))
        except Exception:
            return []


def initialize_chroma(persist_dir: str = None) -> VectorStore:
    """
    Initialize and return a ChromaDB vector store.
    
    Args:
        persist_dir: Directory for persistent storage
        
    Returns:
        Initialized VectorStore object
    """
    return VectorStore(persist_dir)


if __name__ == "__main__":
    # Test the vector store
    vs = initialize_chroma()
    print(f"Document count: {vs.get_document_count()}")
    print(f"Sources: {vs.get_all_sources()}")

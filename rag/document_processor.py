"""
Document processor module for loading and chunking PDF files.
Handles text extraction and splitting with metadata tracking.
"""

from typing import List, Dict, Any
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
import config


def load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    pdf_path = Path(file_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    reader = PdfReader(str(pdf_path))
    text = ""
    
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"[Page {page_num + 1}]\n{page_text}\n\n"
    
    return text.strip()


def split_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into chunks with specified size and overlap.
    Uses recursive character splitting for better chunk boundaries.
    
    Args:
        text: Input text to split
        chunk_size: Target chunk size in characters (default: from config)
        overlap: Overlap between chunks in characters (default: from config)
        
    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if overlap is None:
        overlap = config.CHUNK_OVERLAP
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate end position
        end = start + chunk_size
        
        # If we're at the end, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        
        # Try to find a good breaking point (newline, period, or space)
        chunk_text = text[start:end]
        
        # Look for newline
        last_newline = chunk_text.rfind('\n')
        # Look for period
        last_period = chunk_text.rfind('. ')
        # Look for space
        last_space = chunk_text.rfind(' ')
        
        # Choose the best breaking point
        break_point = max(last_newline, last_period, last_space)
        
        if break_point > chunk_size * 0.5:  # At least halfway through chunk
            end = start + break_point + 1
            chunk_text = text[start:end].strip()
        
        chunks.append(chunk_text)
        start = end - overlap  # Move with overlap
    
    return chunks


def process_document(file_path: str, priority: str = "General", 
                    doc_type: str = None) -> List[Document]:
    """
    Process a PDF document: extract text, split into chunks, and create Document objects.
    
    Args:
        file_path: Path to the PDF file
        priority: Priority level ("Bible", "Notes", "General")
        doc_type: Type of document (defaults to priority value)
        
    Returns:
        List of LangChain Document objects with metadata
    """
    from datetime import datetime
    
    # Load PDF text
    text = load_pdf(file_path)
    
    # Split into chunks
    chunks = split_text(text)
    
    # Create Document objects
    documents = []
    pdf_path = Path(file_path)
    
    if doc_type is None:
        doc_type = priority
    
    for idx, chunk in enumerate(chunks):
        if chunk.strip():  # Skip empty chunks
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": pdf_path.name,
                    "full_path": str(pdf_path.absolute()),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "priority": config.PRIORITY_MAP.get(priority, "low"),
                    "doc_type": doc_type,
                    "timestamp": datetime.now().isoformat(),
                    "chunk_size": len(chunk)
                }
            )
            documents.append(doc)
    
    return documents


def get_all_pdf_files(directory: str = None) -> List[Path]:
    """
    Get all PDF files from a directory.
    
    Args:
        directory: Directory path (defaults to DOCUMENTS_DIR from config)
        
    Returns:
        List of Path objects for PDF files
    """
    if directory is None:
        directory = config.DOCUMENTS_DIR
    
    dir_path = Path(directory)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        return []
    
    return list(dir_path.glob("*.pdf"))


if __name__ == "__main__":
    # Test the document processor
    test_pdf = config.DEFAULT_DOCUMENT_PATH
    if test_pdf.exists():
        print(f"Processing: {test_pdf}")
        docs = process_document(str(test_pdf))
        print(f"Created {len(docs)} document chunks")
        print(f"\nFirst chunk preview:")
        print(docs[0].page_content[:200])
        print(f"\nMetadata: {docs[0].metadata}")
    else:
        print(f"Test file not found: {test_pdf}")

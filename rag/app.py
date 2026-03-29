"""
Gradio web interface for the RAG chatbot.
Features a clean black-themed minimalist design with full response control.
"""

import warnings
import gradio as gr
from pathlib import Path
import config
from rag_pipeline import create_pipeline, RAGPipeline
import os

# Suppress Pydantic V1 deprecation warning (not critical)
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")


# Global pipeline instance (lazy loaded)
pipeline = None


def get_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline instance."""
    global pipeline
    if pipeline is None:
        pipeline = create_pipeline()
    return pipeline


def format_chat_history(history: list) -> list:
    """Format chat history for display."""
    formatted = []
    for user_msg, assistant_msg in history:
        formatted.append((user_msg, {"text": assistant_msg, "color": "primary"}))
    return formatted


def respond(message: str, history: list, tone: str, length: str, 
            include_references: bool, priority_filter: str) -> list:
    """
    Process user message and generate response.
    
    Args:
        message: User input
        history: Chat history (list of messages)
        tone: Response tone setting
        length: Response length setting
        include_references: Include references flag
        priority_filter: Document priority filter
        
    Returns:
        Updated chat history
    """
    if not message.strip():
        return history
    
    # Get pipeline
    try:
        rag = get_pipeline()
    except Exception as e:
        error_msg = f"Error initializing system: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history
    
    # Generate response
    try:
        # Convert priority filter to proper format
        priority = None if priority_filter == "All Documents" else priority_filter
        
        result = rag.query(
            question=message,
            tone=tone,
            length=length,
            include_references=include_references,
            priority_filter=priority
        )
        
        answer = result["answer"]
        
        # Add source info if available
        if result.get("sources"):
            sources_str = ", ".join(result["sources"])
            answer += f"\n\n📚 Sources: {sources_str}"
        
        if result.get("retrieved_chunks", 0) > 0:
            answer += f"\n\nℹ️ Retrieved {result['retrieved_chunks']} relevant sections"
        
    except Exception as e:
        answer = f"Error processing query: {str(e)}"
    
    # Update history with new format
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history


def add_document(file_path, priority: str) -> str:
    """
    Add a new document to the knowledge base.
    
    Args:
        file_path: Uploaded file path
        priority: Document priority level
        
    Returns:
        Status message
    """
    if file_path is None:
        return "⚠ No file selected"
    
    try:
        rag = get_pipeline()
        
        # Determine doc_type from priority
        doc_type = priority
        
        # Add document
        chunks = rag.add_document(file_path, priority=priority, doc_type=doc_type)
        
        if chunks > 0:
            return f"✓ Added {chunks} chunks from {Path(file_path).name}"
        else:
            return "⚠ Failed to process document"
            
    except Exception as e:
        return f"✗ Error: {str(e)}"


def reindex_documents(priority: str) -> str:
    """
    Re-index all documents in the documents folder.
    
    Args:
        priority: Default priority for documents without specific type
        
    Returns:
        Status message
    """
    try:
        rag = get_pipeline()
        
        # Clear existing index
        rag.clear_knowledge_base()
        
        # Index all documents
        total = rag.index_all_documents()
        
        if total > 0:
            return f"✓ Re-indexed all documents ({total} chunks)"
        else:
            return "⚠ No documents found to index"
            
    except Exception as e:
        return f"✗ Error: {str(e)}"


def load_initial_document() -> str:
    """Load the default document if it exists."""
    try:
        rag = get_pipeline()
        
        if config.DEFAULT_DOCUMENT_PATH.exists():
            chunks = rag.add_document(
                str(config.DEFAULT_DOCUMENT_PATH), 
                priority="General"
            )
            return f"Loaded initial document: {chunks} chunks"
        else:
            return "No default document found"
    except Exception as e:
        return f"Warning: {str(e)}"


def get_stats() -> str:
    """Get knowledge base statistics."""
    try:
        rag = get_pipeline()
        stats = rag.get_stats()
        
        msg = f"📊 Knowledge Base:\n"
        msg += f"- Total chunks: {stats['total_chunks']}\n"
        msg += f"- Loaded documents: {stats['loaded_documents']}\n"
        
        if stats['sources']:
            msg += f"- Sources: {', '.join(stats['sources'])}"
        
        return msg
    except Exception as e:
        return f"Error getting stats: {str(e)}"


# Custom CSS for black theme
custom_css = """
.gradio-container {
    background-color: #000000;
    color: #ffffff;
}

.dark .gradio-container {
    background-color: #000000;
}

.chat-container {
    background-color: #000000;
}

.chatbot {
    background-color: #000000 !important;
    border: 1px solid #333333;
}

.message {
    border-radius: 15px !important;
}

.message.user {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
}

.message.bot {
    background-color: #2d2d2d !important;
    color: #ffffff !important;
}

.gr-button {
    border-radius: 8px;
}

.gr-button-primary {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: none;
}

.gr-button-primary:hover {
    background-color: #e0e0e0 !important;
}

.gr-box {
    border-radius: 10px !important;
    border: 1px solid #333333 !important;
}

.gr-input, .gr-textbox {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #333333 !important;
}

.gr-podcast {
    background-color: #000000 !important;
}

header {
    background-color: #000000 !important;
}

footer {
    background-color: #000000 !important;
}
"""


def create_ui() -> gr.Blocks:
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="Private RAG Chatbot",
        fill_height=True
    ) as demo:
        
        # Header
        gr.Markdown(
            "# 📚 Pistos - Your Private Document Assistant",
            elem_classes=["header"],
            show_label=False
        )
        gr.Markdown(
            "*Ask questions about your documents. Answers are based ONLY on uploaded content.*"
        )
        
        # Main interface
        with gr.Row(equal_height=True):
            # Left column - Chat interface
            with gr.Column(scale=3):
                # Chat display
                chatbot = gr.Chatbot(
                    height=500,
                    avatar_images=(None, "E:\\rag\\image.png"),
                    elem_classes=["chatbot"],
                    show_label=False,
                    placeholder="Ask me anything about your documents...",
                    label="Pistos"
                )
                
                # Input area
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your question here...",
                        scale=4,
                        container=False,
                        show_label=False,
                        elem_classes=["input-box"]
                    )
                    submit_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1,
                        elem_classes=["submit-btn"]
                    )
                
                # Settings accordion
                with gr.Accordion("⚙️ Response Settings", open=False):
                    with gr.Row():
                        tone_radio = gr.Radio(
                            choices=["Simple", "Formal"],
                            value="Simple",
                            label="Tone"
                        )
                        length_radio = gr.Radio(
                            choices=["Short", "Detailed"],
                            value="Short",
                            label="Length"
                        )
                    
                    references_check = gr.Checkbox(
                        label="Include references if possible",
                        value=False
                    )
                    
                    priority_dropdown = gr.Dropdown(
                        choices=["All Documents", "Bible", "Notes", "General"],
                        value="All Documents",
                        label="Document Priority Filter"
                    )
            
            # Right column - Document management
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Documents")
                
                # File upload
                file_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                with gr.Row():
                    priority_select = gr.Dropdown(
                        choices=["Bible", "Notes", "General"],
                        value="General",
                        label="Priority",
                        scale=2
                    )
                    add_doc_btn = gr.Button(
                        "Add",
                        variant="primary",
                        scale=1
                    )
                
                # Management buttons
                reindex_btn = gr.Button(
                    "🔄 Re-index All Documents",
                    variant="secondary"
                )
                
                # Status display
                status_box = gr.Textbox(
                    label="Status",
                    lines=3,
                    max_lines=5,
                    interactive=False
                )
                
                # Stats button
                stats_btn = gr.Button(
                    "📊 Show Stats",
                    variant="secondary"
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown(
            "**Privacy Notice:** All processing happens locally. Your documents never leave this machine."
        )
        
        # Event handlers
        # Submit message
        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, tone_radio, length_radio, 
                   references_check, priority_dropdown],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        )
        
        # Enter key to submit
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, tone_radio, length_radio,
                   references_check, priority_dropdown],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        )
        
        # Add document
        add_doc_btn.click(
            fn=add_document,
            inputs=[file_upload, priority_select],
            outputs=[status_box]
        )
        
        # Re-index
        reindex_btn.click(
            fn=reindex_documents,
            inputs=[priority_select],
            outputs=[status_box]
        )
        
        # Show stats
        stats_btn.click(
            fn=get_stats,
            outputs=[status_box]
        )
        
        # Load initial document on startup
        demo.load(
            fn=load_initial_document,
            outputs=[status_box]
        )
    
    return demo


def main():
    """Main entry point for the application."""
    print("=" * 60)
    print("Starting Private RAG Chatbot...")
    print("=" * 60)
    
    # Ensure documents directory exists
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline WITHOUT loading any documents
    print("\nInitializing RAG pipeline...")
    global pipeline
    from rag_pipeline import create_pipeline
    pipeline = create_pipeline()
    
    print("\n✓ Application ready!")
    print("=" * 60)
    print("🌐 OPEN YOUR BROWSER TO:")
    print("   http://localhost:7860")
    print("=" * 60)
    print("\n📊 Knowledge Base Status:")
    print(f"   Total chunks: {len(pipeline.vector_store.collection.get()['ids']) if pipeline else 0}")
    print(f"   Loaded documents: {len(pipeline.loaded_documents) if pipeline else 0}")
    print("\n💡 Upload PDFs using the 'Upload PDF' button in the interface")
    print("   Press CTRL+C to exit\n")
    
    # Create UI
    demo = create_ui()
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public link
        show_error=True,
        quiet=False,
        theme=gr.themes.Base(),
        css=custom_css
    )


if __name__ == "__main__":
    main()

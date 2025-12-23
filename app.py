import streamlit as st
import tempfile
import os

from rag.loader import load_pdf
from rag.splitter import split_documents
from rag.embeddings import create_or_load_faiss
from rag.retriever import retrieve_context
from rag.generator import generate_answer


def format_answer(answer):
    """Format the answer for better readability and structure"""
    if "I cannot find the answer" in answer:
        return f"❌ {answer}"
    
    # Ensure proper markdown formatting
    formatted = answer.strip()
    
    # Add spacing around lists if needed
    lines = formatted.split('\n')
    result = []
    prev_was_list = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_list_item = (stripped and (
            (stripped[0].isdigit() and len(stripped) > 1 and stripped[1] in ['.', ')']) or
            stripped.startswith('•') or stripped.startswith('-') or stripped.startswith('*')
        ))
        
        if is_list_item and not prev_was_list and result and result[-1].strip():
            result.append("")
        elif not is_list_item and prev_was_list:
            result.append("")
        
        result.append(line)
        prev_was_list = is_list_item
    
    return '\n'.join(result)


st.set_page_config(page_title="Local RAG Document Q&A", layout="wide")


st.title("📄 Document Q&A Assistant (Local RAG)")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="file_uploader")
    
    if uploaded_file:
        # Check if this is a new file
        file_hash = hash(uploaded_file.getvalue())
        is_new_file = st.session_state.current_file_hash != file_hash
        
        if is_new_file:
            # Process new document
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            try:
                with st.spinner("Processing document..."):
                    docs = load_pdf(pdf_path)
                    chunks = split_documents(docs)
                    # Force new index creation for new document
                    st.session_state.vectorstore = create_or_load_faiss(chunks, force_new=True)
                    st.session_state.document_processed = True
                    st.session_state.current_file_hash = file_hash
                    # Clear chat history when new document is loaded
                    st.session_state.chat_history = []
                
                # Clean up temp file
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
                
                st.success("✅ Document processed successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Ensure Ollama is running and 'mistral' model is installed.")
                st.session_state.document_processed = False
        else:
            if st.session_state.document_processed:
                st.success("✅ Document ready")
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# Main chat interface
if st.session_state.document_processed and st.session_state.vectorstore:
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])
    
    # Query input at the bottom (always visible)
    st.divider()
    
    with st.form("query_form", clear_on_submit=True):
        query = st.text_input(
            "💬 Ask a question about the document:",
            placeholder="Type your question here...",
            key=f"query_input_{len(st.session_state.chat_history)}"
        )
        submit_button = st.form_submit_button("Send", use_container_width=True)
    
    if submit_button and query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        try:
            with st.spinner("🔍 Retrieving relevant chunks..."):
                context_docs = retrieve_context(query, st.session_state.vectorstore, k=5)
            with st.spinner("🤖 Generating answer..."):
                answer = generate_answer(context_docs, query)
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": format_answer(answer)
            })
            st.rerun()
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}\n\nPlease ensure Ollama is running."
            })
            st.rerun()

else:
    # No document uploaded
    st.info("👆 **Please upload a PDF document in the sidebar to get started.**")
    
    # Show instructions
    with st.expander("📖 How to use"):
        st.markdown("""
        1. **Upload a PDF** using the sidebar on the left
        2. Wait for the document to be processed
        3. **Ask questions** in the chat interface
        4. View your **conversation history** as you chat
        5. Use **Clear Chat History** to start fresh
        """)
    
    if st.session_state.document_processed:
        # Reset state when no file is uploaded
        st.session_state.document_processed = False
        st.session_state.vectorstore = None
        st.session_state.current_file_hash = None
        st.session_state.chat_history = []

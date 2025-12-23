import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


INDEX_PATH = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_PATH, "index.faiss")


def create_or_load_faiss(chunks, force_new=False):
    """
    Creates a new FAISS index from chunks or loads existing one.
    
    Args:
        chunks: List of document chunks to embed
        force_new: If True, always create a new index (default: False)
    
    Returns:
        FAISS vectorstore
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create index directory if it doesn't exist
    os.makedirs(INDEX_PATH, exist_ok=True)

    # If force_new is True or no index exists, create a new one
    if force_new or not os.path.exists(INDEX_FILE):
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)
        return vectorstore

    # Otherwise load existing index
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

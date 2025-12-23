from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Splits documents into manageable overlapping chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks

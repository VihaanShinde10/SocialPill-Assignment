def retrieve_context(query, vectorstore, k=5, score_threshold=0.25):
    """Retrieves top-k relevant chunks from FAISS with similarity filtering"""
    # Get more candidates for better context
    results = vectorstore.similarity_search_with_score(query, k=k*2)
    
    filtered_docs = []
    for doc, score in results:
        similarity = 1 / (1 + score)
        if similarity >= score_threshold:
            filtered_docs.append(doc)
        if len(filtered_docs) >= k:
            break
    
    # Return filtered docs or top k if filtering was too strict
    return filtered_docs if len(filtered_docs) >= k else [doc for doc, _ in results[:k]]

from langchain_community.llms import Ollama
import hashlib

_llm_instance = None
_query_cache = {}


def get_llm():
    """Get or create Ollama LLM instance (singleton for performance)"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = Ollama(model="mistral", temperature=0.2, num_predict=1000)
    return _llm_instance


def _get_query_hash(query, context_text):
    """Generate hash for query+context for caching"""
    combined = f"{query.lower().strip()}:{context_text[:500]}"
    return hashlib.md5(combined.encode()).hexdigest()


def truncate_context(context_text, max_chars=3000):
    """Truncate context to max_chars while preserving complete chunks"""
    if len(context_text) <= max_chars:
        return context_text
    
    # Try to preserve complete chunks by truncating at boundaries
    chunks = context_text.split("\n\n")
    result = []
    current_len = 0
    
    for chunk in chunks:
        if current_len + len(chunk) <= max_chars:
            result.append(chunk)
            current_len += len(chunk) + 2  # +2 for "\n\n"
        else:
            break
    
    return "\n\n".join(result) if result else context_text[:max_chars]


def generate_answer(context_docs, query, use_cache=True):
    """Generates well-structured answer strictly from retrieved context"""
    if not context_docs:
        return "I cannot find the answer to that question in the provided document."

    # Build context with chunk references for better structure
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        content = doc.page_content.strip()
        context_parts.append(f"[Section {i}]\n{content}")
    
    context_text = "\n\n".join(context_parts)
    
    if use_cache:
        cache_key = _get_query_hash(query, context_text)
        if cache_key in _query_cache:
            return _query_cache[cache_key]
    
    context_text = truncate_context(context_text, max_chars=3000)

    # Enhanced prompt for better structured answers
    prompt = f"""You are a helpful document Q&A assistant. Answer the question using ONLY the information from the context provided below.

        INSTRUCTIONS:
        1. Provide a clear, well-structured answer based on the context
        2. Use proper paragraphs and formatting
        3. If the answer involves multiple points, organize them clearly (use bullet points or numbered lists if appropriate)
        4. Be specific and cite relevant details from the context
        5. If the answer cannot be found in the context, respond with: "I cannot find the answer to that question in the provided document."
        6. Do NOT use any information outside the provided context
        7. Make your answer comprehensive but concise

        CONTEXT FROM DOCUMENT:
        {context_text}

        QUESTION: {query}

        Provide a well-structured answer:"""

    try:
        response = get_llm().invoke(prompt).strip()
        
        # Clean up response
        if not response or len(response) < 10:
            response = "I cannot find the answer to that question in the provided document."
        
        # Post-process to improve structure
        response = _improve_answer_structure(response)
        
        if use_cache:
            cache_key = _get_query_hash(query, context_text)
            _query_cache[cache_key] = response
            if len(_query_cache) > 100:
                _query_cache.pop(next(iter(_query_cache)))
        
        return response
    except Exception as e:
        return f"Error: Unable to generate answer. Please ensure Ollama is running. ({str(e)})"


def _improve_answer_structure(answer):
    """Post-process answer to improve structure and readability"""
    if "I cannot find the answer" in answer:
        return answer
    
    lines = answer.split('\n')
    improved = []
    in_list = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                in_list = False
            improved.append("")
            continue
        
        # Detect and format lists
        if (stripped[0].isdigit() and len(stripped) > 1 and stripped[1] in ['.', ')']) or \
           stripped.startswith('-') or stripped.startswith('*'):
            if not in_list:
                improved.append("")
            prefix = "• " if not stripped[0].isdigit() else ""
            improved.append(prefix + stripped)
            in_list = True
        else:
            if in_list:
                improved.append("")
                in_list = False
            improved.append(line)
    
    return '\n'.join(improved).strip()

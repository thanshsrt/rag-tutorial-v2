import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from get_embedding_function import get_embedding_function
import time

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

print("🔧 Initializing RAG components (one-time)...")
_init_start = time.time()

EMBEDDING_FUNCTION = get_embedding_function()
DB = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_FUNCTION)

CHAT_MODEL = ChatOllama(model="llama3.2:3b", temperature=0.1)
print(f"✅ Initialization complete in {time.time() - _init_start:.2f}s")

print("🔥 Warming up LLM...")
_warmup_start = time.time()
_ = CHAT_MODEL.invoke("Hi")
print(f"✅ Warmup complete in {time.time() - _warmup_start:.2f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)

def query_rag(query_text: str):
    """Non-streaming version for CLI."""
    results = DB.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    response = CHAT_MODEL.invoke(prompt)
    response_text = response.content
    
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

def query_rag_stream(query_text: str):
    """Generator that yields tokens from the LLM."""
    search_start = time.time()
    results = DB.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"⏱️  Retrieval: {time.time() - search_start:.3f}s")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print(f"🔥 Starting LLM stream...")
    t2 = time.time()
    
    stream = CHAT_MODEL.stream(prompt)
    
    try:
        first_chunk = next(stream)
        t3 = time.time()
        print(f"⏱️  TTFT: {t3-t2:.3f}s")
        yield first_chunk.content
        
        chunk_count = 1
        for chunk in stream:
            yield chunk.content
            chunk_count += 1
        
        t4 = time.time()
        print(f"⏱️  Total stream: {t4-t2:.2f}s | Chunks: {chunk_count} | Throughput: {chunk_count/(t4-t2):.1f} tok/s")
        print(f"📄 Sources: {sources}")
    except StopIteration:
        print("⚠️  No response from model")

def query_rag_with_metadata(query_text: str):
    """
    Enhanced version with raw_chunks, confidence, and sources.
    Used by /query-json and /query-enriched endpoints.
    """
    results = DB.similarity_search_with_score(query_text, k=5)
    avg_score = sum(score for _, score in results) / len(results) if results else 0.0
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    
    raw_chunks = [
        {
            "content": doc.page_content[:600],  # Slightly larger for agents
            "score": float(score),
            "file": doc.metadata.get("id", "unknown")
        }
        for doc, score in results
    ]
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    response = CHAT_MODEL.invoke(prompt)
    response_text = response.content
    
    return {
        "answer": response_text,
        "sources": [doc.metadata.get("id") for doc, _ in results],
        "raw_chunks": raw_chunks,
        "confidence": round(float(avg_score), 2),
        "query": query_text
    }

if __name__ == "__main__":
    main()
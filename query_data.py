import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function
import time

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# 🚀 SINGLETON INITIALIZATION - happens once at module load
print("🔧 Initializing RAG components (one-time)...")
_init_start = time.time()

EMBEDDING_FUNCTION = get_embedding_function()
DB = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDING_FUNCTION)
# MODEL = Ollama(model="mistral")
# MODEL = Ollama(model="qwen2.5:3b")
MODEL = Ollama(model="phi3:mini")

print(f"✅ Initialization complete in {time.time() - _init_start:.2f}s")

# Warm up model with dummy call
print("🔥 Warming up LLM...")
_warmup_start = time.time()
_ = MODEL.invoke("Hi")  # Force model load into VRAM
print(f"✅ Warmup complete in {time.time() - _warmup_start:.2f}s")

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    """Non-streaming version for CLI."""
    # Prepare the DB.
    # embedding_function = get_embedding_function()
    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = DB.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # model = Ollama(model="mistral")
    response_text = MODEL.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

def query_rag_stream(query_text: str):
    """
    Generator that yields tokens from the LLM after retrieving relevant context.
    Optimized: Uses pre-loaded singletons.
    """
    # Prepare the DB (same as query_rag)
    # embedding_function = get_embedding_function()
    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    search_start = time.time()
    results = DB.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"⏱️  Retrieval: {time.time() - search_start:.3f}s")

    # Build prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Use streaming LLM
    # model = Ollama(model="mistral")
    print(f"🔥 Starting LLM stream...")
    t2 = time.time()
    
    stream = MODEL.stream(prompt)
    
    try:
        first_chunk = next(stream)
        t3 = time.time()
        print(f"⏱️  TTFT (Time to First Token): {t3-t2:.3f}s")
        yield first_chunk
        
        # Continue streaming rest
        chunk_count = 1
        for chunk in stream:
            yield chunk
            chunk_count += 1
    # for chunk in MODEL.stream(prompt):
    #     yield chunk
    #     full_response.append(chunk)
    
        t4 = time.time()
        print(f"⏱️  Total stream time: {t4-t2:.2f}s | Chunks: {chunk_count} | Throughput: {chunk_count/(t4-t2):.1f} tok/s")
        print(f"📄 Sources: {sources}")
    except StopIteration:
        print("⚠️  No response from model")
        
def get_db():
    return DB

if __name__ == "__main__":
    main()

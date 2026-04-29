from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

import time
import asyncio

from hybrid_search import HybridRetriever

from query_data import query_rag_stream, query_rag_with_metadata
from query_data import DB
from query_data import CHAT_MODEL

from server_agent import build_pr_review_graph

app = FastAPI()

# ========== INITIALIZE HYBRID RETRIEVER ==========
print("🔧 Initializing hybrid search...")
try:
    # Get all documents from your existing Chroma DB
    db_data = DB.get(include=["documents", "metadatas"])
    all_docs = db_data["documents"]
    all_metadatas = db_data.get("metadatas")
    hybrid_retriever = HybridRetriever(DB, all_docs, all_metadatas, k=5)

    print("🧠 Compiling LangGraph...")
    agent_app = build_pr_review_graph(CHAT_MODEL, hybrid_retriever)
    print(f"✅ Hybrid search ready: {len(all_docs)} documents indexed")
except Exception as e:
    print(f"⚠️  Hybrid search init failed: {e}")
    hybrid_retriever = None
# ================================================

# Allow CORS for your Next.js app (which runs on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    
class HybridRequest(BaseModel):
    question: str
    alpha: float = 0.5
    candidates: int = 100
    
class PRReviewRequest(BaseModel):
    pr_url: str
    
SYSTEM_PROMPT = """You are a Senior Software Architect performing a Code Review.

INPUT PROVIDED:
1. The PR Diff (The proposed changes)
2. Internal Context (How we currently do things — may be empty or irrelevant)

ABSOLUTE RULES:
- NEVER invent code snippets, diff lines, or variable names that are not explicitly in the PR Diff or Internal Context provided above.
- If the INTERNAL CODEBASE CONTEXT says "No matching internal patterns found" or is unrelated to the PR's language, IGNORE it. Review based solely on the PR diff.
- Only compare against internal patterns if the internal files are clearly the SAME codebase or SAME language.
- NEVER repeat the same title for multiple observations. Each observation must have a unique, specific title.
- Provide exactly 3 technical observations.
- For each observation, provide:
  - TITLE: A unique, specific issue or praise.
  - CONTEXT: Evidence from the provided diff only. Do not invent examples.
  - ACTION: A clear instruction for the developer.

Be brief, direct, and omit conversational filler."""

async def generate_stream(question: str):
    """Convert the synchronous generator to an async generator for FastAPI."""
    for token in query_rag_stream(question):
        yield f"data: {token}\n\n"
        await asyncio.sleep(0)  # yield control to event loop

@app.middleware("http")
async def add_timing_headers(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"⏱️  {request.method} {request.url.path} took {duration:.2f}s")
    response.headers["X-Response-Time"] = str(duration)
    return response

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    return StreamingResponse(
        generate_stream(request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/query-json")  # For agents who need structured data
async def query_json_endpoint(request: QueryRequest):
    """Non-streaming, full metadata response for agents."""
    result = query_rag_with_metadata(request.question)
    return result
    
@app.post("/retrieve")
async def retrieve_only(request: QueryRequest):
    """Fast retrieval for agents - no LLM generation."""
    start = time.time()
    results = DB.similarity_search_with_score(request.question, k=5)
    chunks = [
        {"text": doc.page_content, "source": doc.metadata.get("id"), "score": float(score)}
        for doc, score in results
    ]
    print(f"⏱️  /retrieve took {time.time() - start:.3f}s")
    return {"chunks": chunks, "query": request.question}

@app.post("/retrieve_hybrid")
async def retrieve_hybrid(
    request: HybridRequest
):
    """
    Hybrid search: BM25 + Vector with candidate capping.
    
    - alpha: 0=keyword only, 1=semantic only, 0.5=balanced
    - candidates: How many docs to retrieve from vector DB (default 100)
    """
    if hybrid_retriever is None:
        return {"error": "Hybrid search not initialized"}, 503
    
    results = hybrid_retriever.search(
        request.question, 
        alpha=request.alpha,          
        candidate_count=request.candidates  
    )
    
    return {
        "chunks": results,
        "query": request.question,
        "method": "hybrid",
        "alpha": request.alpha,
        "candidates_considered": request.candidates
    }
    
@app.post("/review-pr")
async def review_pr_endpoint(request: PRReviewRequest):
    async def stream_generator():
        start_time = time.time()
        yield "data: [Connected. Analyzing PR...]\n\n"
        await asyncio.sleep(0.01)
        
        initial_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Review this PR: {request.pr_url.strip()}")
            ],
            "diff_text": None,
            "rag_sources": None
        }
        
        try:
            full_review = ""
            eval_result = None
            
            async for event in agent_app.astream(initial_state, config={"recursion_limit": 10}):
                for node_name, node_state in event.items():
                    elapsed = time.time() - start_time
                    
                    if "messages" in node_state:
                        last_msg = node_state["messages"][-1]
                        
                        is_final = (
                            node_name == "review"
                            and hasattr(last_msg, "content")
                            and getattr(last_msg, "content", None)
                        )
                        
                        if is_final:
                            safe = last_msg.content.replace("\n", " | ")
                            yield f"data: ✅ REVIEW COMPLETE ({elapsed:.1f}s): {safe}\n\n"
                            
                            if "evaluation" in node_state:
                                eval_result = node_state["evaluation"]
                                status_emoji = "✅" if eval_result["passed"] else "⚠️"
                                yield f"data: {status_emoji} QUALITY: {eval_result['score']}/10 | Passed: {eval_result['passed']}\n\n"
                                
                                # NEW: Warn if domain mismatch
                                if eval_result["metrics"].get("domain_mismatch"):
                                    yield f"data: ⚠️ WARNING: Review claims internal comparison but no matching patterns found. Treat with skepticism.\n\n"
                        else:
                            yield f"data: [Status: {node_name} running... ({elapsed:.1f}s)]\n\n"
                
                await asyncio.sleep(0.05)
                
        except Exception as e:
            import traceback
            err = traceback.format_exc().replace("\n", " | ")
            yield f"data: [SERVER ERROR: {err}]\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
    
@app.post("/query-enriched")
async def query_enriched_endpoint(request: QueryRequest):
    """Returns answer + raw_chunks + confidence for agent consumption."""
    return query_rag_with_metadata(request.question)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time
from fastapi import Request
import asyncio

# Import your streaming function
from query_data import query_rag_stream
from query_data import get_db

app = FastAPI()

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
    
@app.post("/retrieve")
async def retrieve_only(request: QueryRequest):
    """Fast retrieval for agents - no LLM generation."""
    start = time.time()
    db = get_db()
    results = db.similarity_search_with_score(request.question, k=5)
    chunks = [
        {"text": doc.page_content, "source": doc.metadata.get("id"), "score": float(score)}
        for doc, score in results
    ]
    print(f"⏱️  /retrieve took {time.time() - start:.3f}s")
    return {"chunks": chunks, "query": request.question}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
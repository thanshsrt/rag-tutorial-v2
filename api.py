from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

# Import your streaming function
from query_data import query_rag_stream

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

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
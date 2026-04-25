import requests

query = "digest authentication implementation"

# Test pure BM25 (alpha in JSON body)
bm25_resp = requests.post(
    "http://localhost:5000/retrieve_hybrid",
    json={"question": query, "alpha": 0.0, "candidates": 100}
)
print(f"BM25 Status: {bm25_resp.status_code}")
if bm25_resp.status_code == 200:
    bm25 = bm25_resp.json()
    print(f"BM25 top: {bm25['chunks'][0].get('source', 'N/A') if bm25.get('chunks') else 'No results'}")
else:
    print(f"Error: {bm25_resp.text[:200]}")

# Test pure vector
vector_resp = requests.post(
    "http://localhost:5000/retrieve_hybrid",
    json={"question": query, "alpha": 1.0, "candidates": 100}
)
print(f"\nVector Status: {vector_resp.status_code}")

# Test hybrid
hybrid_resp = requests.post(
    "http://localhost:5000/retrieve_hybrid",
    json={"question": query, "alpha": 0.5, "candidates": 100}
)
print(f"\nHybrid Status: {hybrid_resp.status_code}")
if hybrid_resp.status_code == 200:
    hybrid = hybrid_resp.json()
    print(f"Hybrid top: {hybrid['chunks'][0].get('source', 'N/A') if hybrid.get('chunks') else 'No results'}")
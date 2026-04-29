import re
import os
import requests
from urllib.parse import urlparse
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from evaluation import evaluate_review, log_evaluation

class State(TypedDict):
    """LangGraph state with messages, diff text, and RAG sources."""
    messages: Annotated[list, add_messages]
    diff_text: Optional[str]
    rag_sources: Optional[list]

def _split_camel_case(name: str) -> str:
    """Convert camelCase/PascalCase to space-separated lowercase words."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()

def _extract_search_queries(diff_text: str, max_queries: int = 3) -> str:
    """Extract search keywords from any PR diff using regex. No LLM involved."""
    if not diff_text or len(diff_text) < 20 or "Error" in diff_text[:20]:
        return "general code patterns"
    
    queries = []
    
    # File paths from +++ headers
    files = re.findall(r'(?:\+\+\+) [ab]/(.+?)(?:\t|\n)', diff_text)
    for f in files[:2]:
        no_ext = re.sub(r'\.\w+$', '', f)
        parts = no_ext.replace('/', ' ').replace('_', ' ').split()
        noise = {'src', 'test', 'tests', 'docs', 'doc', 'lib', 'app', 
                'main', 'init', 'utils', 'helpers', 'common', 'core',
                'babel', 'plugin', 'plugins'}
        meaningful = [p for p in parts if p not in noise and len(p) > 2]
        if meaningful:
            split_parts = [_split_camel_case(p) for p in meaningful[-3:]]
            queries.append(' '.join(split_parts))
    
    # Function/class names from changed lines
    names = re.findall(
        r'^[+-].*?\b(?:def|class|func|function|const|let|var|fn|impl|struct|interface|type)\s+(?:\w+\s+)?([\w\d_]+)',
        diff_text, re.MULTILINE
    )
    for n in list(dict.fromkeys(names))[:2]:
        queries.append(_split_camel_case(n))
    
    # Imports/dependencies
    imports = re.findall(
        r'^[+-]\s*(?:from|import|require|use|using)\s+[\'"]?([\w\d\._\-\/@]+)',
        diff_text, re.MULTILINE
    )
    stdlib = {'os', 'sys', 're', 'json', 'time', 'typing', 'collections', 
              'math', 'random', 'datetime', 'itertools', 'functools',
              'warnings', 'base64', 'urllib', 'http', 'pathlib', 'abc',
              'copy', 'enum', 'hashlib', 'inspect', 'io', 'logging'}
    clean_imports = []
    for i in imports:
        top = i.split('/')[0].split('@')[0].split('.')[0]
        if top not in stdlib and len(top) > 2:
            clean_imports.append(top)
    for imp in list(dict.fromkeys(clean_imports))[:1]:
        queries.append(imp.lower())
    
    # Fallback to repo name
    if not queries:
        repo_match = re.search(r'github\.com/[^/]+/([^/]+)', diff_text)
        if repo_match:
            queries.append(repo_match.group(1).lower())
    
    # Deduplicate and cleanup
    seen = set()
    final = []
    for q in queries:
        q = q.strip()
        if q and q not in seen and len(q) > 2:
            seen.add(q)
            final.append(q)
    
    result = ", ".join(final) if final else "code patterns"
    print(f"🔍 Extracted queries: {result}")
    return result

@tool
def fetch_pr_diff(pr_url: str) -> str:
    """Fetch a GitHub pull request diff for review."""
    try:
        parsed = urlparse(pr_url.strip())
        parts = [p for p in parsed.path.split('/') if p]
        if len(parts) < 4:
            return "Error: Invalid GitHub PR URL."
        
        owner, repo, pr_number = parts[0], parts[1], parts[3]
        diff_url = f"https://github.com/{owner}/{repo}/pull/{pr_number}.diff"
        
        headers = {}
        if os.getenv("GITHUB_TOKEN"):
            headers["Authorization"] = f"token {os.getenv('GITHUB_TOKEN')}"
            
        response = requests.get(diff_url, headers=headers, timeout=15)
        if response.status_code != 200:
            return f"Error: GitHub returned status {response.status_code}"
        
        lines = response.text.split('\n')
        clean_lines = []
        skip_prefixes = ('diff --git', 'index ', 'new file mode', 'deleted file mode')
        
        for line in lines:
            if not line.startswith(skip_prefixes):
                clean_lines.append(line)
        
        return "\n".join(clean_lines)[:2000]
    except Exception as e:
        return f"Error fetching PR: {str(e)}"

def create_local_rag_tool(retriever):
    """Factory that creates a RAG search tool bound to the hybrid retriever."""
    
    @tool
    def internal_rag_search(query: str) -> str:
        """Search the internal codebase for existing patterns."""
        if retriever is None:
            return "NOTICE: RAG Retriever not initialized."
        try:
            q_list = [q.strip() for q in query.split(',') if len(q.strip()) > 2]
            all_results = []
            for q in q_list[:2]:
                all_results.extend(retriever.search(q, alpha=0.5, candidate_count=10))
            
            seen = set()
            unique = []
            for r in all_results:
                src = r.get('source', 'unknown')
                if src not in seen:
                    seen.add(src)
                    unique.append(r)
            
            if not unique:
                return "NOTICE: No matching internal patterns found for this PR."
            
            formatted = "\n\n".join([
                f"FILE: {r['source']}\n{r['content'][:400]}" for r in unique[:3]
            ])
            return f"INTERNAL CODEBASE CONTEXT:\n{formatted}"
        except Exception as e:
            return f"NOTICE: RAG search failed: {str(e)}"
    
    return internal_rag_search

def build_pr_review_graph(llm_model, retriever):
    """Build and compile the deterministic PR review LangGraph."""
    rag_tool = create_local_rag_tool(retriever)
    llm_with_tools = llm_model.bind_tools([fetch_pr_diff, rag_tool])

    def fetch_node(state: State):
        """Inject fetch_pr_diff tool call into the message stream."""
        url = ""
        for m in reversed(state["messages"]):
            match = re.search(r'https?://github\.com/\S+', m.content)
            if match:
                url = match.group(0).strip()
                break
        return {"messages": [AIMessage(
            content="",
            tool_calls=[{"name": "fetch_pr_diff", "args": {"pr_url": url}, "id": "fetch_step"}]
        )]}

    def tool_execution_node(state: State):
        """Execute requested tools and stash results for evaluation."""
        last_message = state["messages"][-1]
        tool_map = {"fetch_pr_diff": fetch_pr_diff, "internal_rag_search": rag_tool}
        
        responses = []
        for call in last_message.tool_calls:
            tool_name = call["name"]
            print(f"🤖 Agent executing tool: {tool_name}")
            result = tool_map[tool_name].invoke(call["args"])
            responses.append(ToolMessage(
                content=str(result), name=tool_name, tool_call_id=call["id"]
            ))
        
        # Stash diff and sources for evaluation
        new_state = {"messages": responses}
        for resp in responses:
            if resp.name == "fetch_pr_diff":
                new_state["diff_text"] = resp.content
            elif resp.name == "internal_rag_search":
                new_state["rag_text"] = resp.content  # Stash raw text
                sources = re.findall(r'FILE:\s+([^\n]+)', resp.content)
                new_state["rag_sources"] = sources
                    
        return new_state

    def extraction_node(state: State):
        """Extract keywords from diff and inject RAG search tool call."""
        diff_text = state["messages"][-1].content
        queries = _extract_search_queries(diff_text)
        return {"messages": [AIMessage(
            content="",
            tool_calls=[{"name": "internal_rag_search", "args": {"query": queries}, "id": "rag_step"}]
        )]}

    def review_node(state: State):
        """Generate final review and evaluate quality immediately."""
        response = llm_model.invoke(state["messages"])
        
        # Evaluate
        diff_text = state.get("diff_text", "")
        rag_sources = state.get("rag_sources", [])
        eval_result = evaluate_review(
                        diff_text, 
                        response.content, 
                        rag_sources, 
                        state.get("rag_text", "")
                    )
        log_evaluation(eval_result)
        print(f"📊 Review Score: {eval_result['score']}/10 | Passed: {eval_result['passed']}")
        
        return {
            "messages": [response],
            "evaluation": eval_result
        }

    workflow = StateGraph(State)
    workflow.add_node("fetch", fetch_node)
    workflow.add_node("execute_fetch", tool_execution_node)
    workflow.add_node("extract", extraction_node)
    workflow.add_node("execute_rag", tool_execution_node)
    workflow.add_node("review", review_node)
    
    workflow.add_edge(START, "fetch")
    workflow.add_edge("fetch", "execute_fetch")
    workflow.add_edge("execute_fetch", "extract")
    workflow.add_edge("extract", "execute_rag")
    workflow.add_edge("execute_rag", "review")
    workflow.add_edge("review", END)
    
    return workflow.compile()
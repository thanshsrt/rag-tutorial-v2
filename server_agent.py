from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
import requests
import os
from urllib.parse import urlparse

# 1. Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Define the Tools

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
        
        # 1500 characters is only about 300 words. 
        # Better to truncate by line or focus on the 'hunks' (changes) only.
        lines = response.text.split('\n')
        clean_lines = []
        skip = ('diff --git', 'index ', 'new file mode', 'deleted file mode')
        
        for line in lines:
            if not line.startswith(skip):
                clean_lines.append(line)
        return "\n".join(clean_lines)[:2500]
    except Exception as e:
        return f"Error fetching PR: {str(e)}"

# We use a factory function so we can inject your existing hybrid_retriever
def create_local_rag_tool(retriever):
    @tool
    def internal_rag_search(query: str) -> str:
        """Search the internal codebase for existing patterns."""
        if retriever is None:
            return "Error: RAG Retriever not initialized on server."
            
        try:
            # HUGE WIN: Direct memory call, no HTTP overhead!
            # results = retriever.search(query, alpha=0.5, candidate_count=100)
            
            # Limit to 3 chunks, and 300 chars each. 
            # Total context stays under 1000 tokens.
            results = retriever.search(query, alpha=0.5, candidate_count=10)
            
            if not results:
                return "No relevant code found in the codebase."

            # formatted = "\n\n".join([
            #     f"File: {c['source']}\n{c['content'][:400]}" for c in results
            # ])
            formatted = "\n\n".join([f"File: {c['source']}\n{c['content'][:300]}" for c in results[:3]])
            return f"Found {len(results)} relevant snippets (showing top 3):\n\n{formatted}"
        except Exception as e:
            return f"Error searching codebase: {str(e)}"
            
    return internal_rag_search

# 3. Graph Builder Factory
def build_pr_review_graph(llm_model, retriever):
    """Builds and compiles the LangGraph workflow."""
    rag_tool = create_local_rag_tool(retriever)
    tools = [fetch_pr_diff, rag_tool]
    
    # Bind tools to the model you pass in (Qwen)
    llm_with_tools = llm_model.bind_tools(tools)

    def chatbot_node(state: State):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def tool_node(state: State):
        last_message = state["messages"][-1]
        tool_map = {
            "fetch_pr_diff": fetch_pr_diff,
            "internal_rag_search": rag_tool
        }
        
        print(f"tool_calls:{last_message.tool_calls}")
        
        tool_responses = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"🤖 Server Agent executing tool: {tool_name}")
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
            else:
                result = f"Error: Tool {tool_name} not found."
                
            tool_responses.append(
                ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call["id"])
            )
        return {"messages": tool_responses}

    def should_continue(state: State) -> str:
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    workflow = StateGraph(State)
    workflow.add_node("agent", chatbot_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    return workflow.compile()
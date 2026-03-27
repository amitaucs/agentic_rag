from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = list(state.get("documents", []))

    tavily_results = web_search_tool.invoke({"query": question})
    search_results = tavily_results.get("results", [])
    joined_tavily_result = "\n".join(
        result["content"] for result in search_results if result.get("content")
    )

    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"question": question, "documents": documents}

if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": []})

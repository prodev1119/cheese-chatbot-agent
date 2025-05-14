from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Any

class CheeseAgentState(TypedDict):
    input: str
    results: List[Any]

def build_cheese_agent(hybrid_search):
    def cheese_search_node(state: CheeseAgentState) -> CheeseAgentState:
        query = state["input"]
        results = hybrid_search.search(query)
        return {"input": query, "results": results}

    graph = StateGraph(CheeseAgentState)
    graph.add_node("search", cheese_search_node)
    graph.set_entry_point("search")
    graph.set_finish_point("search")
    return graph.compile()

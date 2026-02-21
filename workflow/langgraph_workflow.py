from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from agents.orchestrator_agent import ProductOrchestratorAgent

class GraphState(TypedDict):
    messages: Annotated[list, "add_message"]
    query: str
    route_result: dict
    final_answer: str


def router_node(state: GraphState, orchestrator: ProductOrchestratorAgent) -> GraphState:
    """Delegates to orchestrator.route_query()"""

    result = orchestrator.route_query(state["query"])
    return {
        "messages": state["messages"] + [AIMessage(content=f"Routed: {result['route']}")],
         "route_result": result
    }

def format_final(state: GraphState) -> GraphState:
    result = state["route_result"]
    final_msg = f"**{result['route'].upper()} RESULT**\n\n{result['result']}"
    return {
        "messages": state["messages"] + [AIMessage(content=final_msg)],
        "final_answer": final_msg
    }

def create_workflow(orchestrator: ProductOrchestratorAgent):
    workflow = StateGraph(GraphState)

    def router_with_orch(state):
        return router_node(state, orchestrator)
    
    workflow.add_node("router", router_with_orch)
    workflow.add_node("formatter", format_final)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "formatter")
    workflow.add_edge("formatter", END)

    return workflow.compile()
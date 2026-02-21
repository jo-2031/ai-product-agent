import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from agents.product_search_agent import ProductCollectionAgent
from agents.orchestrator_agent import ProductOrchestratorAgent
from workflow.langgraph_workflow import create_workflow
from utils.logging import logger
import json

CSV_PATH = "/Users/jothikaravichandran/Documents/Self_Projects/personal_project/ai_geeks_product_agent/ai-product-agent/source_product_input/merged_product_data.csv"

st.set_page_config(page_title="AI Product Agent", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_agent(_reload_key="default"):
    """Cache with reload key to force refresh"""
    logger.info("Loading RAG Agent...")
    rag_agent = ProductCollectionAgent("./chroma_db") 
    rag_agent.load_and_process_data(CSV_PATH)
    orchestrator = ProductOrchestratorAgent(rag_agent)
    workflow = create_workflow(orchestrator)
    logger.info("Agent loaded!")
    return workflow

def main():
    st.title("AI Product Assistant")
    st.markdown("---")
    
    # Workflow Status Display (Always Visible)
    workflow_status = st.sidebar.empty()
    
    with st.sidebar:
        st.header("Agent Control")
        
        # Reload with workflow feedback
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Reload Agent", type="secondary", use_container_width=True):
                with workflow_status.container():
                    workflow_status.info("Reloading agent workflow...")
                    
                    # Show workflow steps
                    workflow_status.status("Step 1: Clearing cache...")
                    st.cache_resource.clear()
                    
                    workflow_status.status("Step 2: Loading new agent...")
                    try:
                        new_workflow = load_agent(f"reload_{st.session_state.get('reload_count', 0)}")
                        workflow_status.success("Step 3: Workflow ready!")
                        st.session_state.reload_count = st.session_state.get('reload_count', 0) + 1
                    except Exception as e:
                        workflow_status.error(f"Reload failed: {e}")
                    
                    st.rerun()
        
        show_flow = st.checkbox("Show Agent Flow", value=True)
        st.markdown(f"Status: Connected | Reloads: {st.session_state.get('reload_count', 0)}")
    
    # Initialize session state (NEVER CLEAR ON RELOAD)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "agent_steps" in message and show_flow:
                with st.expander("Agent Execution Flow", expanded=False):
                    for step in message["agent_steps"]:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"**{step.get('agent', 'Unknown')}**")
                        with col2:
                            st.markdown(f"*{step.get('action', '')}*")
                            tools = step.get("tools", [])
                            if tools:
                                for tool in tools:
                                    st.success(f"Tool: {tool}")
            
            if "route" in message:
                st.caption(f"Route: {message['route'].upper()} | Agent: {message.get('agent_name', 'Orchestrator')}")
    
    prompt = st.chat_input("Ask about products, prices, discounts...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Agent thinking..."):
                try:
                    workflow = load_agent()
                    
                    result = workflow.invoke({
                        "messages": st.session_state.messages[-10:],
                        "query": prompt
                    })
                    
                    final_answer = result.get('final_answer', result.get('route_result', {}).get('result', 'No response'))
                    route = result.get('route_result', {}).get('route', 'general')
                    
                    agent_steps = [
                        {"agent": "OrchestratorAgent", "tools": ["route_classifier"], "action": f"Routed to {route}"},
                        {"agent": "ProductCollectionAgent" if route == "rag" else "GeneralAgent", 
                         "tools": ["retrieve_product_context"] if route == "rag" else [], 
                         "action": "Retrieved products" if route == "rag" else "Direct response"}
                    ]
                    
                    message_placeholder.markdown(final_answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "route": route,
                        "agent_name": "Orchestrator -> " + ("RAG Agent" if route == "rag" else "General"),
                        "agent_steps": agent_steps
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    


if __name__ == "__main__":
    main()

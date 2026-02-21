import streamlit as st
from agents.product_search_agent import ProductCollectionAgent
from agents.orchestrator_agent import ProductOrchestratorAgent
from workflow.langgraph_workflow import create_workflow
from utils.logging import logger
import json
import os


st.set_page_config(page_title="AI Product Agent", 
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )

@st.cache_resource
def load_agent(_csv_path):
    """Load agent once - cached"""
    logger.info("....Loading RAG Agent...")

    rag_agent = ProductCollectionAgent()
    rag_agent.load_and_process_data(_csv_path)

    orchestrator = ProductOrchestratorAgent(rag_agent)
    workflow = create_workflow(orchestrator)

    logger.info("Agent loaded and workflow created successfully!")
    return workflow

def main():
    st.title("🛍️ AI Product Assistant")
    st.markdown("---")

    #sidebar
    with st.sidebar:
        st.header("Settings")
        csv_path = st.text_input("CSV File Path", 
                                 value="/Users/jothikaravichandran/Documents/Self_Projects/personal_project/ai_geeks_product_agent/ai-product-agent/source_product_input/merged_product_data.csv",
                                 help="Enter the path to your product CSV file.")
        
        if st.button("Reload Agent", type="secondary"):
            st.cache_resource.clear()
            st.rerun()

    if "message" not in st.session_state:
        st.session_state.messages = []

    #Display chat
    for message in st.sessiion_state.messages:
        with st.chat_session(message["role"]):
            st.markdown(message["content"])

            if "route" in message:
                st.caption(f"**Route**: {message['route']}")

    
    if prompt := st.chat_input("Ask about products, prices, or anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    workflow = load_agent(csv_path)
                    
                    # Run workflow
                    result = workflow.invoke({
                        "messages": st.session_state.messages[-5:],  # Last 5 messages context
                        "query": prompt
                    })
                    
                    # Extract response
                    final_answer = result.get('final_answer', result.get('route_result', {}).get('result', 'No response'))
                    route = result.get('route_result', {}).get('route', 'general')
                    
                    # Display response
                    st.markdown(final_answer)
                    st.caption(f"**Route**: {route.upper()}")
                    
                    # Store full message with metadata
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_answer,
                        "route": route
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Agent error: {e}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Sorry, I encountered an error: {str(e)}"
                    })
    if st.sidebar.checkbox("Show Debug"):
        st.sidebar.json({
            "Total Messages": len(st.session_state.messages),
            "CSV Loaded": os.path.exists(csv_path),
            "Session State": {k: type(v).__name__ for k, v in st.session_state.items()}
        })


if __name__ == "__main__":
    main()

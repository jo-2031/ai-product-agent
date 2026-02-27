import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from workflow.master_workflow import MasterOrchestrator
from langchain_core.messages import HumanMessage
# from models.product_schema import Product
from utils.logging import logger

CSV_PATH = "/Users/jothikaravichandran/Documents/Self_Projects/personal_project/ai_geeks_product_agent/ai-product-agent/source_product_input/merged_product_data.csv"

st.set_page_config(page_title="AI Product Agent", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better product cards
st.markdown("""
<style>
    /* Product card styling */
    .stImage {
        border-radius: 8px;
        margin-bottom: 10px;
    }
    
    /* Better spacing */
    [data-testid="column"] {
        padding: 10px;
    }
    
    /* Card container */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_workflow(_reload_key="default", _force_rebuild=False):
    """Cache workflow with reload key to force refresh"""
    import shutil
    from pathlib import Path
    
    logger.info("Loading Master Orchestrator Workflow...")
    
    # Only delete ChromaDB if force_rebuild is True (from reload button)
    if _force_rebuild:
        chroma_db_path = Path(__file__).parent / "agents" / "product_collection_rag_agent" / "chroma_db"
        if chroma_db_path.exists():
            logger.info(f"🗑️ Deleting existing ChromaDB at: {chroma_db_path}")
            shutil.rmtree(chroma_db_path)
            logger.info("✅ ChromaDB deleted successfully")
    
    orchestrator = MasterOrchestrator()
    
    # Load product data into search agent (will use existing ChromaDB or create new if deleted)
    logger.info(f"📊 Loading product data from: {CSV_PATH}")
    orchestrator.search_agent.load_and_process_data(CSV_PATH)
    
    logger.info("✅ Workflow loaded!")
    return orchestrator

def main():
    st.title("🛍️ AI Product Assistant")
    st.markdown("Multi-Agent Conversational Product Recommendation System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 Agent Control")
        
        # Reload button
        if st.button("🔄 Reload Workflow", type="secondary"):
            st.cache_resource.clear()
            st.session_state.reload_count = st.session_state.get('reload_count', 0) + 1
            # Force rebuild ChromaDB on next load
            st.session_state.force_rebuild = True
            st.success("Workflow will reload with fresh data!")
            st.rerun()
        
    
        
        st.markdown("---")
        show_workflow = st.checkbox("Show Workflow Steps", value=True)
        st.markdown("**Status:** 🟢 Connected")
    
    # Initialize session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = {
            "conversation_stage": "greeting",
            "current_products": [],
            "saved_products": []
        }
    
    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show product images if available
            if msg["role"] == "assistant" and "products" in msg and msg["products"]:
                products = msg["products"]
                cols = st.columns(3)
                for idx, product_dict in enumerate(products):
                    with cols[idx % 3]:
                        img_url = product_dict.get('Product Image URL', '').strip()
                        if img_url and img_url not in ['N/A', '', 'Not specified']:
                            try:
                                st.image(img_url, caption=f"Product {idx+1}", width="stretch")
                            except:
                                st.caption(f"📦 Product {idx+1}")
            
            # Show workflow steps
            if "workflow_info" in msg and show_workflow and msg["role"] == "assistant":
                with st.expander("🔄 Workflow Execution", expanded=False):
                    workflow_info = msg["workflow_info"]
                    
                    # Show stage
                    st.info(f"**Current Stage:** {workflow_info.get('stage', 'N/A')}")
                    
                    # Show user query
                    if "user_query" in workflow_info and workflow_info['user_query'] != 'N/A':
                        st.success(f"**Query:** {workflow_info['user_query']}")
    
    # Chat input
    prompt = st.chat_input("💬 Say hi or ask: 'Show me laptops under 80000'")
    
    if prompt:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process with workflow
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get force_rebuild flag and reset it
                    force_rebuild = st.session_state.get('force_rebuild', False)
                    if force_rebuild:
                        st.session_state.force_rebuild = False
                    
                    workflow = load_workflow(_force_rebuild=force_rebuild)
                    
                    # Build messages for workflow
                    workflow_messages = [HumanMessage(content=prompt)]
                    
                    # Invoke workflow
                    config = {"configurable": {"thread_id": "streamlit_session"}}
                    result = workflow.graph.invoke(
                        {
                            "messages": workflow_messages,
                            **st.session_state.workflow_state
                        },
                        config
                    )
                    
                    # Extract response
                    response_message = result["messages"][-1]
                    response_text = response_message.content
                    
                    # Update workflow state
                    st.session_state.workflow_state.update({
                        "conversation_stage": result.get("stage", "greeting"),
                        "current_products": result.get("products", []),
                        "user_query": result.get("user_query", prompt)
                    })
                    
                    # Determine which stage
                    stage = result.get("stage", "greeting")
                    
                    stage_map = {
                        "greeting": "1️⃣ Greeting Stage",
                        "search": "2️⃣ Product Search (RAG Agent)",
                        "awaiting_compare": "2️⃣ Product Search - Awaiting Compare",
                        "compare": "3️⃣ Comparison (Multi-Agent)",
                        "awaiting_recommend": "3️⃣ Comparison - Awaiting Recommendation",
                        "recommend": "4️⃣ Recommendation Engine",
                        "memory": "5️⃣ Memory Save",
                        "close": "6️⃣ Continue/Close Decision",
                        "exit": "👋 Goodbye"
                    }
                    
                    stage_display = stage_map.get(stage, f"Stage: {stage}")
                    
                    workflow_info = {
                        "stage": stage_display,
                        "user_query": result.get("user_query", "N/A")
                    }
                    
                    # Display response
                    st.markdown(response_text)
                    
                    # Display product images
                    products = result.get("products", [])
                    if products:
                        cols = st.columns(3)
                        for idx, product_dict in enumerate(products):
                            with cols[idx % 3]:
                                img_url = product_dict.get('Product Image URL', '').strip()
                                if img_url and img_url not in ['N/A', '', 'Not specified']:
                                    try:
                                        st.image(img_url, caption=f"Product {idx+1}", width="stretch")
                                    except:
                                        st.caption(f"📦 Product {idx+1}")
                    
                    # Save to chat history
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "workflow_info": workflow_info,
                        "products": products
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Workflow error: {e}", exc_info=True)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    


if __name__ == "__main__":
    main()

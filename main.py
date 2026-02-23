
"""
AI Product Agent V2 — Interactive Streamlit UI
Handles LangGraph interrupt() resume pattern.
"""
import sys, os, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langgraph.types import Command
from agents.product_search_agent import ProductCollectionAgent
from agents.memory_manager import MemoryManager
from workflow.langgraph_workflow import create_workflow
from utils.logging import logger

CSV_PATH = "/Users/jaigayatiri/Documents/ai-product-agent-main/source_product_input/merged_product_data_clean.csv"

st.set_page_config(page_title="AI Product Agent V2", layout="wide")


# ── LOAD AGENTS ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_agent():
    memory_manager = MemoryManager()
    rag_agent      = ProductCollectionAgent("./chroma_db_v2")
    rag_agent.load_and_process_data(CSV_PATH)
    workflow       = create_workflow(rag_agent, memory_manager)
    return workflow, memory_manager


# ── PRODUCT CARDS ─────────────────────────────────────────────────────────────

def show_scored_cards(products: list):
    if not products:
        return
    cols = st.columns(min(len(products[:5]), 3))
    for i, p in enumerate(products[:5]):
        with cols[i % 3]:
            img = p.get("Product Image URL", "")
            if img and img.startswith("http"):
                st.image(img, use_container_width=True)
            score = p.get("final_score", "N/A")
            st.markdown(f"**#{i+1} — {p.get('Product','N/A')[:45]}**")
            st.caption(f"{p.get('Brands','N/A')} | {p.get('Selling Price','N/A')} | ⭐{p.get('Rating','N/A')}")
            if score != "N/A":
                st.progress(float(score) / 10, text=f"Score: {score}/10")
            st.caption(f"Sentiment: {p.get('sentiment_label','?')} | Brand: {p.get('brand_label','?')}")


# ── MAIN UI ───────────────────────────────────────────────────────────────────

def main():
    st.title("AI Product Assistant V2")
    st.caption("Interactive multi-agent pipeline with step-by-step user input")

    # Session state
    if "messages"    not in st.session_state: st.session_state.messages    = []
    if "session_id"  not in st.session_state: st.session_state.session_id  = str(uuid.uuid4())
    if "interrupted" not in st.session_state: st.session_state.interrupted = False

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("ranked_products"):
                show_scored_cards(msg["ranked_products"])

    # Input hint based on state
    hint = (
        "Respond to the agent's question above..."
        if st.session_state.interrupted
        else "Ask for a recommendation, search, or say hi..."
    )
    prompt = st.chat_input(hint)
    if not prompt:
        return

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    workflow, memory_manager = load_agent()
    config = memory_manager.get_config(st.session_state.session_id)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.interrupted:
                    # Resume graph with user's answer to the interrupt
                    result = workflow.invoke(Command(resume=prompt), config=config)
                else:
                    # Fresh invocation
                    state = {
                        "query"          : prompt,
                        "messages"       : [],
                        "raw_products"   : [],
                        "ranked_products": [],
                        "top_product"    : {},
                        "result"         : "",
                        "final_answer"   : "",
                        "refine_count"   : 0,
                        "conv_context"   : {},
                        "intent"         : "",
                        "brand_filter"   : "",
                        "type_filter"    : "",
                        "show_all"       : False,
                    }
                    result = workflow.invoke(state, config=config)

                # Check if graph is interrupted (waiting for user)
                # LangGraph returns a dict with "__interrupt__" key containing Interrupt objects
                interrupt_list = None
                if isinstance(result, dict) and result.get("__interrupt__"):
                    interrupt_list = result["__interrupt__"]
                elif hasattr(result, "__interrupt__") and result.__interrupt__:
                    interrupt_list = result.__interrupt__

                if interrupt_list:
                    # Extract question text from the Interrupt object(s)
                    first = interrupt_list[0]
                    if hasattr(first, "value"):
                        question = first.value
                    elif isinstance(first, dict):
                        question = first.get("value", str(first))
                    else:
                        question = str(first)

                    st.markdown(question)
                    st.session_state.interrupted = True

                    # Only show product cards at the ask_pick interrupt
                    # (when user is explicitly choosing from the ranked list)
                    ranked = result.get("ranked_products", []) if isinstance(result, dict) else []
                    show_cards = bool(ranked) and "Type **1–" in question
                    if show_cards:
                        show_scored_cards(ranked)

                    st.session_state.messages.append({
                        "role"           : "assistant",
                        "content"        : question,
                        "ranked_products": ranked if show_cards else []
                    })

                else:
                    # Graph completed
                    st.session_state.interrupted = False
                    final_answer = result.get("final_answer") or result.get("result", "No response.")

                    st.markdown(final_answer)
                    st.session_state.messages.append({
                        "role"   : "assistant",
                        "content": final_answer,
                    })

            except Exception as e:
                # LangGraph raises GraphInterrupt — handle it
                exc_type = type(e).__name__
                if "Interrupt" in exc_type or "interrupt" in str(e).lower():
                    # Extract value from Interrupt exception
                    if hasattr(e, "args") and e.args:
                        interrupts = e.args[0]
                        if isinstance(interrupts, (list, tuple)) and interrupts:
                            first = interrupts[0]
                            question = first.value if hasattr(first, "value") else str(first)
                        else:
                            question = str(interrupts)
                    else:
                        question = str(e)
                    st.markdown(question)
                    st.session_state.interrupted = True
                    st.session_state.messages.append({"role": "assistant", "content": question})
                else:
                    st.error(f"Error: {e}")
                    logger.error("UI error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()

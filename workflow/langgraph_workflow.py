"""
Interactive Multi-Agent LangGraph Workflow — V2

Interrupt points:
  1. ask_brand_node  → user provides brand preference
  2. ask_type_node   → user provides type/category preference
  3. ask_show_node   → user chooses: show all ranked results or go straight to recommendation
  4. ask_pick_node   → user picks product by number from the ranked list
  5. confirm_node    → user says YES / NO / feedback
"""
import concurrent.futures
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from agents.product_search_agent import ProductCollectionAgent
from agents.intent_classifier_agent import IntentClassifierAgent
from agents.scoring_agent import ScoringAgent
from agents.sentiment_agent import SentimentAgent
from agents.branding_agent import BrandingAgent
from agents.aggregator_agent import AggregatorAgent
from agents.recommendation_agent import RecommendationAgent
from agents.human_validation_node import HumanValidationNode
from agents.memory_manager import MemoryManager
from utils.logging import logger

MAX_REFINE_LOOPS = 3


# ── STATE ─────────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    query: str
    intent: str
    messages: Annotated[list, "add_message"]
    raw_products: List[Dict[str, Any]]
    ranked_products: List[Dict[str, Any]]
    top_product: Dict[str, Any]
    result: str
    final_answer: str
    refine_count: int
    conv_context: Dict[str, Any]
    brand_filter: str
    type_filter: str
    show_all: bool


# ── NODE 1: INTENT ────────────────────────────────────────────────────────────

def intent_node(state: GraphState, classifier: IntentClassifierAgent) -> dict:
    intent = classifier.classify(state["query"])
    logger.info("IntentNode → %s", intent)
    return {"intent": intent}


# ── NODE 2a: GREET ────────────────────────────────────────────────────────────

def greet_node(state: GraphState, rag_agent: ProductCollectionAgent, llm: ChatOpenAI) -> dict:
    catalog = rag_agent.get_catalog_summary()
    prompt = (
        f"You are an AI product shopping assistant. User said: \"{state['query']}\"\n\n"
        f"Respond with a short friendly greeting. Mention catalog: {catalog}. "
        f"Give 2-3 example queries like: "
        f"'recommend me a Fastrack watch under ₹3000', 'show laptops under ₹50000', "
        f"'compare Boat and Sony earbuds'. Keep it concise."
    )
    answer = llm.invoke([HumanMessage(content=prompt)]).content
    return {"final_answer": answer, "result": answer}


# ── NODE 2b: SEARCH (no scoring) ─────────────────────────────────────────────

def search_node(state: GraphState, rag_agent: ProductCollectionAgent) -> dict:
    query = state["query"]
    products = rag_agent.retrieve_products(query)
    products = rag_agent.filter_by_category(query, products)
    products = rag_agent.filter_by_price(query, products)

    if not products:
        return {"final_answer": "No products found matching your query.", "ranked_products": []}

    lines = [
        f"- {p.get('Product', 'N/A')[:50]} | {p.get('Brands', 'N/A')} | "
        f"{p.get('Selling Price', 'N/A')} | ⭐{p.get('Rating', 'N/A')}"
        for p in products
    ]
    result = "Here are the products I found:\n\n" + "\n".join(lines)
    return {"final_answer": result, "ranked_products": products}


# ── NODE 3: RAG RETRIEVAL ─────────────────────────────────────────────────────

def rag_node(state: GraphState, rag_agent: ProductCollectionAgent) -> dict:
    query = state["query"]
    products = rag_agent.retrieve_products(query)
    products = rag_agent.filter_by_category(query, products)
    products = rag_agent.filter_by_price(query, products)

    if not products:
        msg = f"No products found for '{query}'. We carry: {rag_agent.get_catalog_summary()}."
        return {"raw_products": [], "final_answer": msg}

    logger.info("RAGNode: %d products retrieved", len(products))
    return {
        "raw_products": products,
        "messages": state["messages"] + [
            AIMessage(content=f"🔍 Found **{len(products)} products** matching your query.")
        ]
    }


# ── NODE 4: INTERRUPT 1 — ASK BRAND ──────────────────────────────────────────

def ask_brand_node(state: GraphState) -> dict:
    products = state.get("raw_products", [])
    brands = sorted(set(
        p.get("Brands", "").split()[0] for p in products if p.get("Brands", "").strip()
    ))[:6]
    brand_list = ", ".join(brands) if brands else "various"

    question = (
        f"🔍 I found **{len(products)} products** matching your query.\n"
        f"Available brands: **{brand_list}**\n\n"
        f"Any brand preference? (e.g. {brands[0] if brands else 'any'}) or type **'any'**"
    )
    user_input = interrupt(question)
    return {
        "brand_filter": user_input.strip().lower(),
        "messages": state["messages"] + [AIMessage(content=question)]
    }


# ── NODE 4b: APPLY BRAND FILTER ───────────────────────────────────────────────

def apply_brand_node(state: GraphState) -> dict:
    products = state.get("raw_products", [])
    brand_filter = state.get("brand_filter", "").strip().lower()

    if brand_filter not in ("any", "all", "no", "none", "skip", ""):
        filtered = [p for p in products if brand_filter in p.get("Brands", "").lower()]
        if filtered:
            products = filtered
            logger.info("ApplyBrandNode '%s': %d products", brand_filter, len(products))

    return {"raw_products": products}


# ── NODE 4c: INTERRUPT 2 — ASK TYPE ───────────────────────────────────────────

def ask_type_node(state: GraphState, llm: ChatOpenAI) -> dict:
    products = state.get("raw_products", [])
    brand_filter = state.get("brand_filter", "")
    brand_display = (
        brand_filter.title()
        if brand_filter not in ("any", "all", "no", "none", "skip", "")
        else ""
    )

    product_names = "\n".join(p.get("Product", "") for p in products[:20])
    prompt = f"""Given these product names, list the distinct product types/categories present.
Return a short slash-separated list (e.g. analog / smart / digital). Max 4 types.

Products:
{product_names}

Respond with ONLY the types, nothing else."""

    try:
        type_options = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception:
        type_options = "any"

    label = f"{brand_display} " if brand_display else ""
    question = (
        f"📦 Found **{len(products)} {label}products**.\n"
        f"Want to filter by type? ({type_options} / any)"
    )
    user_input = interrupt(question)
    return {
        "type_filter": user_input.strip().lower(),
        "messages": state["messages"] + [AIMessage(content=question)]
    }


# ── NODE 4d: APPLY TYPE FILTER ────────────────────────────────────────────────

def apply_type_node(state: GraphState) -> dict:
    products = state.get("raw_products", [])
    type_filter = state.get("type_filter", "").strip().lower()

    if type_filter not in ("any", "all", "no", "none", "skip", ""):
        filtered = [p for p in products if type_filter in p.get("Product", "").lower()]
        if filtered:
            products = filtered
            logger.info("ApplyTypeNode '%s': %d products", type_filter, len(products))

    return {"raw_products": products}


# ── NODE 5: PARALLEL SCORING ──────────────────────────────────────────────────

def scoring_node(
    state: GraphState,
    scoring_agent: ScoringAgent,
    sentiment_agent: SentimentAgent,
    branding_agent: BrandingAgent,
    aggregator: AggregatorAgent
) -> dict:
    products = state.get("raw_products", [])
    if not products:
        return {}

    logger.info("ScoringNode: parallel fan-out on %d products", len(products))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f_score = ex.submit(scoring_agent.score, [dict(p) for p in products])
        f_sentiment = ex.submit(sentiment_agent.analyze, [dict(p) for p in products])
        f_brand = ex.submit(branding_agent.analyze, [dict(p) for p in products])
        scored = f_score.result()
        sentiments = f_sentiment.result()
        brands = f_brand.result()

    s_map = {p.get("Product"): p for p in sentiments}
    b_map = {p.get("Product"): p for p in brands}
    for p in scored:
        name = p.get("Product")
        p["sentiment_score"] = s_map.get(name, {}).get("sentiment_score", 5.0)
        p["sentiment_label"] = s_map.get(name, {}).get("sentiment_label", "neutral")
        p["brand_score"] = b_map.get(name, {}).get("brand_score", 5.0)
        p["brand_label"] = b_map.get(name, {}).get("brand_label", "niche")

    ranked = aggregator.aggregate(scored)
    top = ranked[0] if ranked else {}

    summary = "\n".join(
        f"**#{i}** {p.get('Product', '')[:45]} | {p.get('Selling Price', 'N/A')} | "
        f"⭐{p.get('Rating', 'N/A')} | Score: {p.get('final_score', '?')}/10 | "
        f"Sentiment: {p.get('sentiment_label', '?')} | Brand: {p.get('brand_label', '?')}"
        for i, p in enumerate(ranked[:5], 1)
    )

    msg = (
        f"📊 Scored & ranked **{len(ranked)} products**.\n"
        f"Top scorer: **{top.get('Product', '?')[:45]}** "
        f"{top.get('Selling Price', 'N/A')} | Score {top.get('final_score', '?')}/10\n\n"
        f"{summary}"
    )
    return {
        "ranked_products": ranked,
        "top_product": top,
        "messages": state["messages"] + [AIMessage(content=msg)]
    }


# ── NODE 5b: INTERRUPT 3 — ASK SHOW ──────────────────────────────────────────

def ask_show_node(state: GraphState, llm: ChatOpenAI) -> dict:
    ranked = state.get("ranked_products", [])
    n = min(len(ranked), 5)

    question = (
        f"Show all **{n}** ranked results or go straight to recommendation?\n\n"
        f"Type **'all'** to see the full ranked list, or **'go'** to jump to the top pick."
    )
    user_input = interrupt(question)

    prompt = f"""User was asked: "Show all results or go straight to recommendation?"
User replied: "{user_input}"

Did the user want to see all results? Respond with ONLY: yes or no"""

    try:
        show_all = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower() == "yes"
    except Exception:
        show_all = False

    return {
        "show_all": show_all,
        "messages": state["messages"] + [AIMessage(content=question)]
    }


# ── NODE 6: INTERRUPT 4 — ASK PICK ───────────────────────────────────────────

def ask_pick_node(state: GraphState) -> dict:
    ranked = state.get("ranked_products", [])
    n = min(len(ranked), 5)

    lines = "\n".join(
        f"**{i}.** {p.get('Product', '')[:50]} | {p.get('Selling Price', 'N/A')} | "
        f"⭐{p.get('Rating', 'N/A')} | Score: {p.get('final_score', '?')}/10"
        for i, p in enumerate(ranked[:n], 1)
    )
    question = (
        f"Here are the top **{n} matches**:\n\n{lines}\n\n"
        f"Type **1–{n}** to pick a product for a full recommendation, or **'best'** for the top pick."
    )

    user_input = interrupt(question)

    try:
        idx = int(user_input.strip()) - 1
        idx = max(0, min(idx, len(ranked) - 1))
    except ValueError:
        idx = 0

    chosen = ranked[idx] if ranked else {}
    logger.info("PickNode: user picked #%d '%s'", idx + 1, chosen.get("Product", "?"))

    return {
        "top_product": chosen,
        "messages": state["messages"] + [AIMessage(content=question)]
    }


# ── NODE 7: RECOMMENDATION ────────────────────────────────────────────────────

def recommendation_node(state: GraphState, rec_agent: RecommendationAgent) -> dict:
    top = state.get("top_product", {})
    query = state.get("query", "")
    if not top:
        return {"result": "No product selected.", "final_answer": "No product selected."}

    rec = rec_agent.recommend([top], query)
    logger.info("RecommendationNode: '%s'", top.get("Product", "?"))
    return {
        "result": rec["recommendation"],
        "messages": state["messages"] + [AIMessage(content=rec["recommendation"])]
    }


# ── NODE 8: CONFIRM (YES/NO) ──────────────────────────────────────────────────

def confirm_node(state: GraphState, validator: HumanValidationNode, memory_manager: MemoryManager) -> dict:
    result = state.get("result", "")
    top_product = state.get("top_product", {})
    query = state.get("query", "")
    refine_count = state.get("refine_count", 0)

    question = f"{result}\n\n**Would you like to proceed? (yes / no or tell me what to change)**"
    user_input = interrupt(question)

    validation = validator.validate(
        user_response=user_input,
        top_product=top_product,
        original_query=query,
        memory_manager=memory_manager
    )

    if validation["status"] == "confirmed":
        return {
            "final_answer": validation["final_answer"],
            "refine_count": 0,
            "messages": state["messages"] + [AIMessage(content=validation["final_answer"])]
        }
    else:
        new_count = refine_count + 1
        msg = f"🔄 Refining search ({new_count}/{MAX_REFINE_LOOPS})..."
        return {
            "query": validation["refined_query"],
            "refine_count": new_count,
            "final_answer": "",
            "messages": state["messages"] + [AIMessage(content=msg)]
        }


# ── NODE 9: FORMATTER ─────────────────────────────────────────────────────────

def formatter_node(state: GraphState) -> dict:
    intent = state.get("intent", "")
    final_answer = state.get("final_answer") or state.get("result", "No response.")
    formatted = f"**[{intent.upper()}]**\n\n{final_answer}" if intent else final_answer
    return {"final_answer": formatted}


# ── ROUTING ───────────────────────────────────────────────────────────────────

def route_after_intent(state: GraphState) -> str:
    i = state.get("intent", "greet")
    if i == "greet": return "greet"
    if i == "search": return "search"
    return "rag"


def route_after_rag(state: GraphState) -> str:
    return "ask_brand" if state.get("raw_products") else "formatter"


def route_after_ask_show(state: GraphState) -> str:
    return "ask_pick" if state.get("show_all") else "recommend"


def route_after_confirm(state: GraphState) -> str:
    if state.get("final_answer"):
        return "formatter"
    if state.get("refine_count", 0) >= MAX_REFINE_LOOPS:
        return "formatter"
    return "rag"


# ── WORKFLOW FACTORY ──────────────────────────────────────────────────────────

def create_workflow(rag_agent: ProductCollectionAgent, memory_manager: MemoryManager):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    classifier = IntentClassifierAgent()
    scoring_ag = ScoringAgent()
    sentiment_ag = SentimentAgent()
    branding_ag = BrandingAgent(rag_agent._known_brands)
    aggregator_ag = AggregatorAgent()
    rec_ag = RecommendationAgent()
    validator = HumanValidationNode()

    wf = StateGraph(GraphState)

    wf.add_node("intent",      lambda s: intent_node(s, classifier))
    wf.add_node("greet",       lambda s: greet_node(s, rag_agent, llm))
    wf.add_node("search",      lambda s: search_node(s, rag_agent))
    wf.add_node("rag",         lambda s: rag_node(s, rag_agent))
    wf.add_node("ask_brand",   ask_brand_node)
    wf.add_node("apply_brand", apply_brand_node)
    wf.add_node("ask_type",    lambda s: ask_type_node(s, llm))
    wf.add_node("apply_type",  apply_type_node)
    wf.add_node("scoring",     lambda s: scoring_node(s, scoring_ag, sentiment_ag, branding_ag, aggregator_ag))
    wf.add_node("ask_show",    lambda s: ask_show_node(s, llm))
    wf.add_node("ask_pick",    ask_pick_node)
    wf.add_node("recommend",   lambda s: recommendation_node(s, rec_ag))
    wf.add_node("confirm",     lambda s: confirm_node(s, validator, memory_manager))
    wf.add_node("formatter",   formatter_node)

    wf.set_entry_point("intent")

    wf.add_conditional_edges("intent", route_after_intent, {
        "greet": "greet",
        "search": "search",
        "rag": "rag",
    })
    wf.add_edge("greet",  "formatter")
    wf.add_edge("search", "formatter")

    wf.add_conditional_edges("rag", route_after_rag, {
        "ask_brand": "ask_brand",
        "formatter": "formatter",
    })

    wf.add_edge("ask_brand",   "apply_brand")
    wf.add_edge("apply_brand", "ask_type")
    wf.add_edge("ask_type",    "apply_type")
    wf.add_edge("apply_type",  "scoring")

    wf.add_edge("scoring", "ask_show")
    wf.add_conditional_edges("ask_show", route_after_ask_show, {
        "ask_pick": "ask_pick",
        "recommend": "recommend",
    })

    wf.add_edge("ask_pick",  "recommend")
    wf.add_edge("recommend", "confirm")

    wf.add_conditional_edges("confirm", route_after_confirm, {
        "formatter": "formatter",
        "rag": "rag",
    })

    wf.add_edge("formatter", END)

    return wf.compile(checkpointer=memory_manager.get_checkpointer())

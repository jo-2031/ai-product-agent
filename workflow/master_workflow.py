"""
Master Orchestrator - Controls complete conversation flow
Greeting → Search (awaiting_compare) → Compare (awaiting_recommend) → Recommend (awaiting_memory) → Memory (close) → Close
"""
from typing import TypedDict, Literal, Annotated, List
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config.llm_config import LLMConfig
from agents.product_search_agent import ProductSearchAgent
from agents.value_spec_agent import ValueSpecAgent
from agents.value_brand_agent import ValueBrandAgent
from agents.value_sentiment_agent import ValueSentimentAgent
from agents.recommend_agent import RecommendAgent
from utils.logging import logger
import operator

class ConversationState(TypedDict):
    """State for conversation workflow"""
    messages: Annotated[list, operator.add]
    stage: str  # greeting, search, compare, recommend, memory, close, exit
    user_query: str
    products: list  # List of product dicts (converted from Pydantic models)
    products_text: str
    spec_analysis: str
    brand_analysis: str
    sentiment_analysis: str
    user_preferences: dict

class MasterOrchestrator:
    """Master workflow controller"""
    
    def __init__(self):
        """Initialize orchestrator with all agents"""
        self.llm_config = LLMConfig()
        self.llm = self.llm_config.get_chat_model()
        
        # Initialize all agents
        self.search_agent = ProductSearchAgent()
        self.spec_agent = ValueSpecAgent()
        self.brand_agent = ValueBrandAgent()
        self.sentiment_agent = ValueSentimentAgent()
        self.recommend_agent = RecommendAgent()
        
        # Memory for conversation persistence
        self.memory = MemorySaver()
        
        # Build graph
        self.graph = self._build_graph()
        
        logger.info("MasterOrchestrator initialized with all agents")
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("router", self.route_message)
        workflow.add_node("greeting", self.greeting_response)
        workflow.add_node("search", self.search_stage)
        workflow.add_node("compare", self.compare_stage)
        workflow.add_node("recommend", self.recommend_stage)
        workflow.add_node("memory", self.memory_stage)
        workflow.add_node("close", self.close_stage)
        
        # Start with router
        workflow.add_edge(START, "router")
        
        # Router decides where to go
        workflow.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "greeting": "greeting",
                "search": "search",
                "compare": "compare",
                "recommend": "recommend",
                "memory": "memory",
                "close": "close",
                "exit": END
            }
        )
        
        # All paths end after one action (wait for user input)
        workflow.add_edge("greeting", END)
        workflow.add_edge("search", END)
        workflow.add_edge("compare", END)  # Stop after showing comparison
        workflow.add_edge("recommend", END)  # Stop after recommendation
        workflow.add_edge("memory", END)  # Stop after memory
        workflow.add_edge("close", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    # ============ ROUTER NODE ============
    def route_message(self, state: ConversationState):
        """Route incoming message based on current stage and content"""
        messages = state.get("messages", [])
        current_stage = state.get("stage", "greeting")
        user_message = messages[-1].content if messages else ""
        
        logger.info(f"Router: stage={current_stage}, message={user_message}")
        
        # Classify intent based on current stage
        intent_prompt = f"""Classify this user message:

Current conversation stage: {current_stage}
User message: "{user_message}"

Rules:
- If current stage is "greeting" and message is greeting (hi, hello, hey) → "greeting"
- If mentions product/buy/show/laptop/phone/under 70000 → "product_search"
- If current stage is "awaiting_compare" and says compare/yes → "compare"
- If current stage is "awaiting_recommend" and says recommend/yes/which one → "recommend"
- If current stage is "awaiting_memory" and says yes/save → "memory"
- If current stage is "close" and says yes/more → "product_search" (restart)
- If says bye/exit/no thanks → "exit"

Output ONLY ONE WORD: greeting, product_search, compare, recommend, memory, close, or exit"""

        intent_msg = [SystemMessage(content=intent_prompt)]
        intent = self.llm.invoke(intent_msg).content.strip().lower()
        
        logger.info(f"Router decision: {intent}")
        
        return {
            "stage": intent,
            "user_query": user_message if "product_search" in intent else state.get("user_query", "")
        }
    
    def route_decision(self, state: ConversationState) -> Literal["greeting", "search", "compare", "recommend", "memory", "close", "exit"]:
        """Make routing decision"""
        stage = state.get("stage", "greeting")
        
        if "exit" in stage:
            return "exit"
        elif "product_search" in stage or "search" in stage:
            return "search"
        elif "compare" in stage:
            return "compare"
        elif "recommend" in stage:
            return "recommend"
        elif "memory" in stage:
            return "memory"
        elif "close" in stage:
            return "close"
        else:
            return "greeting"
    
    # ============ STAGE 1: GREETING ============
    def greeting_response(self, state: ConversationState):
        """Send greeting message"""
        logger.info("Greeting stage")
        
        greeting_text = """Hi 👋 How can I help you today? \n

You can say things like: \n 
• "I wanted to buy a laptop under 70000" \n 
• "Show me the best watches" \n 
• "Compare the best earbuds" \n
• "I need a phone with good battery" 
"""
        
        return {
            "messages": [AIMessage(content=greeting_text)],
            "stage": "greeting"
        }
    
    # ============ STAGE 2: PRODUCT SEARCH ============
    def search_stage(self, state: ConversationState):
        """Search and show top 3 products"""
        user_query = state.get("user_query", "")
        
        logger.info(f"Search stage: {user_query}")
        
        # Optimized single retrieval call
        products_data = self.search_agent.get_products_data(user_query)
        logger.info(f"Retrieved {len(products_data)} products")
        
        # Check if no products found
        if not products_data or len(products_data) == 0:
            no_products_response = """Sorry, this product is currently not available. \n

                    Please try: \n
                    • Searching with different keywords \n
                    • Using broader search terms  \n
                    • Exploring another product category"""
            
            return {
                "messages": [AIMessage(content=no_products_response)],
                "stage": "greeting",
                "products": [],
                "products_text": no_products_response
            }
        
        # Build product info for LLM
        products_info_text = ""
        for idx, p_dict in enumerate(products_data, 1):
            products_info_text += f"\nProduct {idx}:\n"
            for key, value in p_dict.items():
                if value and value != 'N/A' and key != 'Product ID':
                    products_info_text += f"  {key}: {value}\n"
        
        # LLM formats the response
        llm_prompt = f"""You are a helpful shopping assistant. Here are the top 3 products:

{products_info_text}

Format as a clean list with bullet points for each product showing: Name, Brand, Price, Discount, Rating.
Then ask: "Would you like me to compare these products? Say yes or compare."
"""
        
        response_with_prompt = self.llm.invoke(llm_prompt).content
        
        return {
            "messages": [AIMessage(content=response_with_prompt)],
            "stage": "awaiting_compare",
            "products": products_data,
            "products_text": response_with_prompt
        }
    
    # ============ STAGE 3: COMPARISON ============
    def compare_stage(self, state: ConversationState):
        """Compare products using 3 specialist agents"""
        products_text = state.get("products_text", "")
        
        logger.info("Compare stage: Running 3 specialist agents...")
        
        # Call SpecAgent
        spec_analysis = self.spec_agent.analyze(products_text)
        logger.info("✅ Spec analysis complete")
        
        # Call BrandAgent
        brand_analysis = self.brand_agent.analyze(products_text)
        logger.info("✅ Brand analysis complete")
        
        # Call SentimentAgent
        sentiment_analysis = self.sentiment_agent.analyze(products_text)
        logger.info("✅ Sentiment analysis complete")
        
        # Create comprehensive comparison report
        comparison = f"""🔄 **Comprehensive Product Comparison:**

📊 **1. Value Analysis (Price, Discount, Rating)**
{spec_analysis}

---

🏷️ **2. Brand Analysis**
{brand_analysis}

---

⭐ **3. Sentiment Analysis (Customer Preference)**
{sentiment_analysis}

---

💡 **Ready for my recommendation?**

Say "recommend" or "yes" to see which product I suggest!"""
        
        return {
            "messages": [AIMessage(content=comparison)],
            "stage": "awaiting_recommend",
            "spec_analysis": spec_analysis,
            "brand_analysis": brand_analysis,
            "sentiment_analysis": sentiment_analysis
        }
    
    # ============ STAGE 4: RECOMMENDATION ============
    def recommend_stage(self, state: ConversationState):
        """Make final recommendation"""
        logger.info("Recommend stage: Generating final recommendation...")
        
        spec_analysis = state.get("spec_analysis", "")
        brand_analysis = state.get("brand_analysis", "")
        sentiment_analysis = state.get("sentiment_analysis", "")
        products_text = state.get("products_text", "")
        
        # Call recommend agent
        recommendation = self.recommend_agent.recommend(
            spec_analysis,
            brand_analysis,
            sentiment_analysis,
            products_text
        )
        
        # Add prompt to ask about saving preferences
        final_response = f"""{recommendation}

---

💾 **Would you like me to remember your preferences?**

This will help me give better recommendations next time!

Say "yes" to save, or "no" to skip."""
        
        return {
            "messages": [AIMessage(content=final_response)],
            "stage": "awaiting_memory"
        }
    
    # ============ STAGE 5: MEMORY ============
    def memory_stage(self, state: ConversationState):
        """Save user preferences"""
        logger.info("Memory stage: Saving preferences...")
        
        # Here you can implement actual preference saving logic
        # For now, just acknowledge and move to close
        memory_response = """✅ **Preferences saved!**

I'll remember your choices for next time.

---

🛍️ **Would you like to explore another product?**

Say:
• "Yes" - to search for more products
• "No" / "Exit" / "Bye" - to close"""
        
        return {
            "messages": [AIMessage(content=memory_response)],
            "stage": "close"
        }
    
    # ============ STAGE 6: CLOSE ============
    def close_stage(self, state: ConversationState):
        """Ask to continue or exit"""
        logger.info("Close stage")
        
        close_prompt = """🛍️ **Would you like to explore another product?**

Say:
• "Yes" - to search for more products
• "No" / "Exit" / "Bye" - to close"""
        
        return {
            "messages": [AIMessage(content=close_prompt)],
            "stage": "close"
        }

def create_workflow():
    """Factory function for LangGraph"""
    orchestrator = MasterOrchestrator()
    return orchestrator.graph

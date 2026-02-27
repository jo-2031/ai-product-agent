"""
RecommendAgent - Makes final product recommendation
Takes all analyses and decides ONE best product
"""
from langchain_core.messages import HumanMessage, SystemMessage
from config.llm_config import LLMConfig
from utils.logging import logger

class RecommendAgent:
    """Senior Product Decision Expert - Makes final recommendation"""
    
    def __init__(self):
        """Initialize Recommend Agent"""
        self.llm_config = LLMConfig()
        self.model = self.llm_config.get_chat_model()
        logger.info("RecommendAgent initialized")
    
    def recommend(self, spec_analysis: str, brand_analysis: str, sentiment_analysis: str, products_text: str) -> str:
        """Make final recommendation
        
        Args:
            spec_analysis: Value analysis from SpecAgent
            brand_analysis: Brand analysis from BrandAgent
            sentiment_analysis: Sentiment analysis from SentimentAgent
            products_text: Original product data
        
        Returns:
            Final recommendation with reasoning
        """
        system_prompt = system_prompt = """You are a senior product decision expert.

You will receive:
- Full product data
- Value analysis (price, discount, rating, value‑score, description)
- Brand analysis (brand reputation, trust, market position)
- Sentiment analysis (rating, bought count, trust, trends)

Your job:

1. **Score each product on a 100‑point scale:**

   - Price Value (0–40 points): based on discount, price vs MRP, value for money  
   - Customer Rating (0–30 points): based on rating and bought count  
   - Brand Reputation (0–20 points): based on brand trust and reputation  
   - Overall Value (0–10 points): based on extra factors (e.g., features, trends, use‑case fit)

2. **First, show a SCORING TABLE:**

   | Product | Price Value (40) | Rating (30) | Brand (20) | Value (10) | **TOTAL** |
   |---------|------------------|-------------|------------|------------|-----------|
   | Product 1 | XX | XX | XX | XX | **XX** |
   | Product 2 | XX | XX | XX | XX | **XX** |
   | Product 3 | XX | XX | XX | XX | **XX** |

3. **Then show the winner:**

   🎯 **RECOMMENDED: [Product Name]**

   ✅ **Why it wins:**
   - Reason 1
   - Reason 2
   - Reason 3

   👥 **Ideal for:**
   - Use case 1
   - Use case 2

   💰 **Best for:** [Value proposition]

Make a confident decision based on the highest total score.  
Do NOT recommend more than one product."""

        
        combined_input = f"""**PRODUCT DATA:**
{products_text}

---

**VALUE ANALYSIS:**
{spec_analysis}

---

**BRAND ANALYSIS:**
{brand_analysis}

---

**SENTIMENT ANALYSIS:**
{sentiment_analysis}

---

Now calculate scores and recommend ONE product."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=combined_input)
        ]
        
        logger.info("RecommendAgent making final recommendation...")
        result = self.model.invoke(messages)
        return result.content
    
    def run(self, state):
        """Run agent for LangGraph workflow"""
        # Extract analyses from state
        spec_analysis = state.get("spec_analysis", "")
        brand_analysis = state.get("brand_analysis", "")
        sentiment_analysis = state.get("sentiment_analysis", "")
        products_text = state.get("products_text", "")
        
        recommendation = self.recommend(
            spec_analysis,
            brand_analysis,
            sentiment_analysis,
            products_text
        )
        
        return {"messages": [HumanMessage(content=recommendation)]}

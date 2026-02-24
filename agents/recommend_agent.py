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
        system_prompt = """You are a senior product decision expert.

You will receive 3 analyses. Calculate a score for EACH product.

**SCORING SYSTEM:**
- Price Value (0-40 points): Based on discount, price vs MRP, value for money
- Customer Rating (0-30 points): Based on rating and bought count
- Brand Reputation (0-20 points): Based on brand trust and reputation
- Overall Value (0-10 points): Additional factors

**Total: 100 points**

**OUTPUT FORMAT:**

1. First, show SCORING TABLE:

| Product | Price Value (40) | Rating (30) | Brand (20) | Value (10) | **TOTAL** |
|---------|------------------|-------------|------------|------------|-----------|
| Product 1 | XX | XX | XX | XX | **XX** |
| Product 2 | XX | XX | XX | XX | **XX** |
| Product 3 | XX | XX | XX | XX | **XX** |

2. Then show winner:

🎯 **RECOMMENDED: [Product Name]**

✅ **Why it wins:**
- [Reason 1]
- [Reason 2]
- [Reason 3]

👥 **Ideal for:**
- [Use case 1]
- [Use case 2]

💰 **Best for:** [Value proposition]

Make confident decision based on highest score."""
        
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

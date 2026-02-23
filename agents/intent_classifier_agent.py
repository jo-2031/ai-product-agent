
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.logging import logger

INTENT_LABELS = ["search", "compare", "recommend", "refine", "greet"]

INTENT_DESCRIPTIONS = {
    "search"   : 'looking for products (e.g. "show me laptops under 50000")',
    "compare"  : 'comparing products (e.g. "compare Samsung vs Apple phones")',
    "recommend": 'asking for best/top pick (e.g. "what should I buy?", "best phone under 30000")',
    "refine"   : 'narrowing previous results (e.g. "only show gaming ones", "filter by 4 star")',
    "greet"    : 'greetings or off-topic (e.g. "hi", "hello", "how are you")',
}


class IntentClassifierAgent:
    """Classifies user query into one of 5 intents."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def classify(self, query: str) -> str:
        intent_list = "\n".join(f"- {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items())
        prompt = f"""You are an intent classifier for a product shopping assistant.

Classify the user query into exactly one of these intents:
{intent_list}

User query: "{query}"

Respond with ONLY one word from: {", ".join(INTENT_LABELS)}"""

        intent = self.llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        if intent not in INTENT_LABELS:
            logger.warning("Unexpected intent '%s' — defaulting to 'search'", intent)
            intent = "search"
        logger.info("IntentClassifier: '%s' → %s", query, intent)
        return intent

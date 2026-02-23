import json
import re
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from agents.branding_agent import BrandingAgent
from utils.logging import logger
from utils.price import parse_price

CSV_FIELDS = [
    "Product ID", "Product", "Brands", "Discount",
    "Rating", "Bought Count", "MRP Price", "Selling Price",
    "Delivery Date", "Product Image URL"
]


class ProductCollectionAgent:
    """RAG Agent — loads CSV into ChromaDB, retrieves products via similarity search."""

    def __init__(self, persist_directory: str = "./chroma_db_v2", model_name: str = "gpt-4o-mini"):
        load_dotenv()
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
        self.vector_store = Chroma(
            collection_name="product_collection_v2",
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )
        self._known_brands = set()
        self._csv_path = None

    def _llm_parse_query(self, query: str) -> dict:
        """Use LLM to extract category, brand intent, and price limit from query."""
        prompt = f"""Analyze this product search query and extract structured information.

Query: "{query}"

Respond in JSON only:
{{
  "category": "<product category or null>",
  "is_specific_brand": <true if query mentions a specific unknown brand, else false>,
  "price_limit": <number or null>
}}"""
        try:
            return json.loads(self.llm.invoke([HumanMessage(content=prompt)]).content.strip())
        except (json.JSONDecodeError, ValueError):
            return {"category": None, "is_specific_brand": False, "price_limit": None}

    def load_and_process_data(self, file_path: str):
        self._csv_path = file_path
        self._known_brands = BrandingAgent.load_known_brands(file_path)

        if self.vector_store._collection.count() > 0:
            logger.info("ChromaDB already loaded — skipping reload")
            return

        docs = CSVLoader(
            file_path=file_path,
            csv_args={"delimiter": ",", "quotechar": '"', "fieldnames": CSV_FIELDS},
        ).load()
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        self.vector_store.add_documents(splits)
        logger.info("Loaded %d chunks into ChromaDB", len(splits))

    def retrieve_products(self, query: str, k: int = 30) -> list:
        query_words = set(query.lower().split())
        brand_mentions = query_words & self._known_brands

        if len(brand_mentions) > 1:
            products, seen_ids = [], set()
            for brand in brand_mentions:
                for p in self._docs_to_products(self.vector_store.similarity_search(f"{brand} {query}", k=5)):
                    uid = p.get("Product ID") or p.get("Product", "")
                    if uid and uid not in seen_ids and brand in (p.get("Brands", "") + p.get("Product", "")).lower():
                        seen_ids.add(uid)
                        products.append(p)
        else:
            products = self._docs_to_products(self.vector_store.similarity_search(query, k=k))
            if brand_mentions:
                products = [p for p in products if any(b in p.get("Brands", "").lower() for b in brand_mentions)]
            else:
                parsed = self._llm_parse_query(query)
                if parsed.get("is_specific_brand"):
                    products = []

        logger.info("RAG: %d products for '%s'", len(products), query)
        return products

    def filter_by_category(self, query: str, products: list) -> list:
        parsed = self._llm_parse_query(query)
        category = parsed.get("category")
        if not category:
            return products
        filtered = [p for p in products if category.lower() in p.get("Product", "").lower()]
        return filtered if filtered else products

    def filter_by_price(self, query: str, products: list) -> list:
        match = re.search(r'(?:under|below|less than|within|upto|up to)\s+[₹rs.]?\s*(\d[\d,]*)', query.lower())
        if not match:
            return products
        limit = float(match.group(1).replace(",", ""))
        filtered = [p for p in products if (v := parse_price(p.get("Selling Price", ""))) is not None and v <= limit]
        logger.info("Price filter ≤%d: %d → %d", int(limit), len(products), len(filtered))
        return filtered

    def get_catalog_summary(self) -> str:
        prompt = """You are a product catalog assistant. Summarize the types of products available in one short sentence.
Respond with just the summary, no extra text."""
        try:
            return self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception:
            return "our product catalog"

    def _docs_to_products(self, docs: list) -> list:
        products, seen = [], set()
        for doc in docs:
            product = {
                k.strip(): v.strip()
                for line in doc.page_content.splitlines()
                if ": " in line
                for k, _, v in [line.partition(": ")]
            }
            uid = product.get("Product ID") or product.get("Product", "")
            if uid and uid not in seen and product.get("Brands", "").lower() != "sponsored":
                seen.add(uid)
                products.append(product)
        return products

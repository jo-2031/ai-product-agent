# AI Product Recommendation Agent

A multi‑agent AI system that helps users choose the best e‑commerce product by analyzing **price, brand, and customer sentiment**, then recommending **one optimal product** with clear scoring and provider (Amazon / Flipkart).

## Project Structure

```text
ai-product-agent/
├── .env
├── .env.example
├── .gitignore
├── main.py                         
├── requirements.txt                
├── README.md               
├── agents/
│   ├── __init__.py
│   ├── product_search_agent.py
│   ├── recommend_agent.py
│   ├── value_brand_agent.py
│   ├── value_sentiment_agent.py
├── workflow/
│   ├── __init__.py
│   └── master_workflow.py
├── config/
│   ├── __init__.py
│   ├── llm_config.py
├── utils/
│   ├── logging.py
│   └── __pycache__/
├── source_product_input/
│   └── merged_product_data.csv
└── chroma_db/                       
```

## Prerequisites

- Python 3.10+
- `pip`

## Setup (Create venv + Install requirements)

From the `ai-product-agent` folder:

```bash
# 1) Create virtual environment
python3 -m venv venv

# 2) Activate virtual environment (macOS/Linux)
source venv/bin/activate

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Variables

This project supports both OpenAI and Groq chat models via `config/llm_config.py`.

Create a `.env` file in the project root (same level as `main.py`) and configure one provider:

```env
# Choose provider: openai or groq
LLM_PROVIDER=openai

# Model (example)
LLM_MODEL=gpt-4o-mini

OPENAI_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here
```

Groq example:

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-70b-versatile
GROQ_API_KEY=your_key_here
```

Note: embeddings are configured with OpenAI embeddings in `llm_config.py`.

## Run Streamlit App

```bash
streamlit run main.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Notes

- Product input CSV is loaded from `source_product_input/merged_product_data.csv`.
- If dependency issues appear, ensure venv is activated before running commands.
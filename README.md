# Free RAG Chatbot: Gemini + Local Embeddings

This project demonstrates a **100% Free** implementation of a RAG (Retrieval-Augmented Generation) chatbot. It combines the power of **Google Gemini** (Cloud) for reasoning with **HuggingFace** (Local) for secure and cost-effective data embedding.

## Key Features
- **Zero Cost Architecture:** Utilizes Google Gemini Free Tier and Open Source Local Embeddings.
- **Privacy-Focused Embeddings:** Document vectors are generated locally using `sentence-transformers`, not sent to paid APIs.
- **Performance:** Powered by `Gemini-flash-latest` for fast responses.

## Tech Stack
- **LLM:** Google Gemini Flash via `models/gemini-flash-latest`
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local CPU)
- **Vector DB:** ChromaDB
- **Framework:** LangChain & Streamlit

## Setup
#### 1. Clone the repo & Install dependencies:
    pip install -r requirements.txt
#### 2. Get your FREE API Key from Google AI Studio.
#### 3. Create .env file:
    GOOGLE_API_KEY=your_key_here
#### 4. Place your PDF in data/ folder.
#### 5. Run ingestion (First time run will download the embedding model ~80MB):
    python ingest.py
#### 6. Start Chatbot:
    streamlit run app.py

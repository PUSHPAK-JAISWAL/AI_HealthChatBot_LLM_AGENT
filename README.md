# HealthAlert — AI-Driven Public Health Chatbot (LLM Agent)

HealthAlert is an AI-powered public health chatbot designed to assist citizens—especially in rural and semi-urban areas—with preventive healthcare, disease symptoms, vaccination schedules, and timely outbreak alerts. It combines Retrieval-Augmented Generation (RAG), trusted health web search, and document ingestion with a Large Language Model (LLM) backend.

## Features

- **Conversational AI Chatbot:** Provides actionable, concise health advice tailored to users.
- **Document Ingestion:** Supports storing and searching health documents using [ChromaDB](https://www.trychroma.com/) or an in-memory fallback.
- **RAG + LLM Chat Endpoint:** Integrates document retrieval and web search results for context-aware responses.
- **Trusted Health Web Search:** Queries health advisories from official domains like WHO and Indian government health sites.
- **Outbreak Alert Scanning:** Automatically scans for outbreaks, classifies severity, and saves alerts to MongoDB.
- **Session Management:** Handles multi-user sessions and token-aware trimming for long conversations, with optional summarization.
- **Extensible Storage:** Uses MongoDB for persistent storage of users, sessions, documents, and alerts.

## Tech Stack

- **Backend:** Python, [FastAPI](https://fastapi.tiangolo.com/)
- **Database:** [MongoDB](https://www.mongodb.com/) (via [Motor](https://motor.readthedocs.io/))
- **Embeddings:** [SentenceTransformers](https://www.sbert.net/)
- **Vector DB:** [ChromaDB](https://www.trychroma.com/) (optional)
- **LLM:** [Ollama](https://ollama.com/) (default model: `gemma3:4b`)
- **Web Search:** [Google Search API](https://pypi.org/project/googlesearch-python/) (optional)
- **Token Counting:** [tiktoken](https://github.com/openai/tiktoken) (optional)

## Getting Started

### Prerequisites

- Python 3.8+
- [MongoDB](https://www.mongodb.com/try/download/community)
- (Optional) [ChromaDB](https://github.com/chroma-core/chroma)
- (Optional) [Ollama](https://ollama.com/)
- (Optional) [tiktoken](https://github.com/openai/tiktoken)
- (Optional) [googlesearch-python](https://pypi.org/project/googlesearch-python/)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/PUSHPAK-JAISWAL/AI_HealthChatBot_LLM_AGENT.git
    cd AI_HealthChatBot_LLM_AGENT
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**
    Create a `.env` file in the root directory, or use system environment variables. Example:
    ```
    MONGO_URI=mongodb://localhost:27017
    DB_NAME=health_alert
    OLLAMA_URL=http://localhost:11434
    OLLAMA_MODEL=gemma3:4b
    CHROMA_PERSIST_DIR=./chromadb_persist
    SESSION_CONTEXT_MAX_TOKENS=128000
    SUMMARIZE_OLD_MESSAGES=false
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    ```

4. **Run the server:**
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

## API Overview

All endpoints are served from `/`, e.g. `http://localhost:8000/`.

### User & Session Management

- `POST /user` — Create or update a user
- `GET /user/{user_id}` — Retrieve user profile
- `DELETE /user/{user_id}` — Delete user and their sessions
- `POST /session` — Create a chat session
- `GET /session/{session_id}` — Retrieve session details
- `DELETE /session/{session_id}` — Delete session

### Chat & Retrieval Endpoints

- `POST /session/{session_id}/message` — Send a message, receive AI response (RAG + LLM)
- `POST /docs` — Ingest a health document (title, content, source)

### Outbreak Alerts

- `GET /alerts/scan` — On-demand region health scan (triggers web + doc search, classifies outbreak)
- `GET /alerts/latest` — Get latest stored alert for a region

### Misc

- `GET /health` — Health check and configuration overview

## How It Works

- **Chatbot**: Uses FastAPI for API endpoints. Messages are stored in sessions, trimmed to a configurable token budget. Optionally summarizes older messages using LLM.
- **RAG**: Combines embeddings-based document retrieval (ChromaDB or in-memory) and trusted web search results to provide grounded answers.
- **Alerts**: Scans for outbreaks using web search and local docs, leverages LLM to classify severity, saves results to MongoDB.

## Customization

- **Trusted Domains:** Configurable in `HEALTH_DOMAIN_WHITELIST` in `app/main.py`.
- **Embedding Model:** Set via `EMBEDDING_MODEL` in `.env`.
- **LLM Model:** Set via `OLLAMA_MODEL` in `.env`.
- **Session Token Limit:** Adjust `SESSION_CONTEXT_MAX_TOKENS` in `.env`.

## Example Usage

1. Ingest a document:
    ```bash
    curl -X POST http://localhost:8000/docs -H "Content-Type: application/json" \
      -d '{"title":"COVID-19 Advisory","content":"Wear masks in public places...","source":"who.int"}'
    ```

2. Start a chat session:
    ```bash
    curl -X POST http://localhost:8000/session -H "Content-Type: application/json" \
      -d '{"user_id":"user123"}'
    ```

3. Send a message:
    ```bash
    curl -X POST http://localhost:8000/session/{session_id}/message -H "Content-Type: application/json" \
      -d '{"user_id":"user123","text":"What are the symptoms of dengue?"}'
    ```

4. Trigger an alert scan:
    ```bash
    curl http://localhost:8000/alerts/scan?region=India
    ```

## Troubleshooting

- **ChromaDB not installed:** Falls back to in-memory document retrieval.
- **Ollama not running:** LLM features will fail; ensure Ollama server is up and accessible.
- **Google Search API missing:** Web search features will be unavailable.

## License

MIT License

## Author

Created by [Pushpak Jaiswal](https://github.com/PUSHPAK-JAISWAL)

---

*For further documentation, see the source code in [`app/main.py`](app/main.py).*

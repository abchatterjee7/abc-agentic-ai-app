# ABC Agentic AI Assistant (FastAPI + Streamlit + LangGraph + ChromaDB + PostgreSQL)

A dual-capability this agentic AI app:

1. **RAG** – answers questions about company policies (leave, benefits, remote work, expenses, conduct) from a ChromaDB vector store.
2. **Tool execution** – checks and cancels orders in a PostgreSQL database using LangChain tools.

Everything runs in Docker, so you only need Docker Desktop and a **free Google AI Studio API key** (Gemini models, no billing required).

## Architecture

```
Streamlit (8501) ──HTTP──> FastAPI (8000) ──> LangGraph agent ──> Google Gemini (LLM + embeddings)
                                                  │
                                   ┌──────────────┴──────────────┐
                              ChromaDB (8001)             PostgreSQL (5432)
                          (policy embeddings)               (orders table)
```

LangGraph flow:

```
START → intent_classifier ─┬─ rag  → rag_node ─────────────┐
                           ├─ tool → tool_agent ⇄ tools ───┼→ response_generator → END
                           └─ general ─────────────────────┘
```

* `intent_classifier` – LLM (structured output) picks `rag`, `tool` or `general`, using recent history for follow-ups like "cancel it".
* `rag_node` – embeds the question, retrieves top-k chunks from ChromaDB, records sources.
* `tool_agent` / `tools` – LLM bound to `get_order_status` and `cancel_order`; loops until no more tool calls.
* `response_generator` – writes the final answer (grounded in retrieved context for RAG).
* Conversation memory is kept per `thread_id` (LangGraph `MemorySaver`, in-memory: cleared when the backend restarts).

## Project structure

```
abc-agentic-ai-app/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app & endpoints
│   │   ├── config.py          # Settings from env / .env
│   │   ├── schemas.py         # Pydantic models
│   │   ├── bootstrap.py       # Startup: wait for services, create tables, seed, ingest
│   │   ├── cli.py             # Manual seed / ingest commands
│   │   ├── db/                # SQLAlchemy engine, Order model, seeding
│   │   ├── agents/            # LangGraph state + graph
│   │   ├── tools/             # @tool definitions
│   │   └── vectorstore/       # ChromaDB client, ingestion
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                 # Streamlit UI
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   ├── policies/*.txt         # Sample policy documents (edit / add your own)
│   └── seed_orders.json       # Sample orders
├── docker-compose.yml
├── .env.example
└── README.md
```

## Quick start (Windows 11)

### 1. Prerequisites

* **Docker Desktop for Windows** (with the WSL 2 backend enabled, the default): https://www.docker.com/products/docker-desktop/
  Start Docker Desktop and wait until it says "Engine running".
* A free **Google AI Studio API key**: https://aistudio.google.com/apikey

### 2. Configure

Open **PowerShell** in the project folder (`abc-agentic-ai-app`):

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `GOOGLE_API_KEY=...` and save.

### 3. Start everything

```powershell
docker compose up --build
```

The first build takes a few minutes. On startup the backend automatically:

* creates the `orders` table and seeds 8 sample orders,
* chunks, embeds and stores the policy files in ChromaDB.

Watch for `Application startup complete` in the backend logs.

### 4. Use it

| Service | URL |
|---|---|
| Streamlit chat UI | http://localhost:8501 |
| FastAPI docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/api/v1/health |

Try in the chat:

* *How many days of annual leave do I get?* → RAG (sources shown)
* *Can I work from another country?* → RAG
* *What is the status of order ORD-101?* → `get_order_status`
* *Cancel order ORD-1004* → `cancel_order` (allowed: confirmed)
* *Cancel order ORD-1002* → refused (already shipped)
* *Cancel it* after an order question → follow-up resolved from history

Sample order IDs: `ORD-1001` … `ORD-1008` (statuses: processing, shipped, delivered, confirmed, pending, cancelled, processing, shipped).

### 5. Stop / restart

```powershell
docker compose stop          # stop, keep data
docker compose up -d         # start again in background
docker compose down          # remove containers, keep data volumes
docker compose down -v       # remove containers AND all data (fresh start)
```

## API

**POST `/api/v1/chat`**

```json
{ "message": "Cancel order ORD-1004", "thread_id": "abc123", "model": "gemini-3.5-flash" }
```

Response:

```json
{ "response": "Order ORD-1004 has been cancelled successfully.", "tool_used": "cancel_order", "sources": [] }
```

`model` is optional (must be in `AVAILABLE_MODELS`). Reuse the same `thread_id` to keep conversation context.

**GET `/api/v1/health`** – reports database, vector store and API-key status. **GET `/api/v1/models`** – lists selectable models.

PowerShell test:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/v1/chat -Method Post -ContentType "application/json" `
  -Body '{"message":"What is the status of order ORD-101?","thread_id":"t1"}'
```

## Common tasks

**Add or edit policy documents** – put `.txt` files in `data/policies/` (first line: `Title: Your Title`), then rebuild the index:

```powershell
docker compose build backend
docker compose up -d backend
docker compose exec backend python -m app.cli ingest --force
```

**Reset sample orders**

```powershell
docker compose exec backend python -m app.cli seed --reset
```

**View logs**

```powershell
docker compose logs -f backend
```

**Inspect the database**

```powershell
docker compose exec postgres psql -U agent -d agentdb -c "SELECT order_id, status FROM orders;"
```

**Change the Google API key or models** – edit `.env`, then `docker compose up -d` (recreates the backend).

## Running without Docker for the app (optional, for development)

Keep Postgres and Chroma in Docker, run the code locally (Python 3.11):

```powershell
docker compose up -d postgres chroma

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt -r frontend\requirements.txt

cd backend
uvicorn app.main:app --reload --port 8000
```

In a second PowerShell window (venv activated):

```powershell
cd frontend
streamlit run app.py
```

Local defaults point at `localhost:5432` (Postgres) and `localhost:8001` (Chroma), matching the published Docker ports.

## Troubleshooting

* **"error during connect" / Docker not running** – start Docker Desktop and retry.
* **Port already in use** (8000, 8001, 8501, 5432) – stop the other program or change the left-hand port in `docker-compose.yml` (e.g. `"8502:8501"`).
* **Sidebar shows ❌ Gemini API key** – `GOOGLE_API_KEY` is empty in `.env`; edit it and run `docker compose up -d`.
* **`429` / quota errors** – you hit the free-tier rate limit (requests per minute/day differ per model). Wait a minute, or pick a `flash-lite` model in the sidebar, which usually has higher limits. Each chat turn makes 2–3 model calls.
* **`404 model not found`** – that model name isn't available to your key; choose another in the sidebar or edit `AVAILABLE_MODELS` in `.env`.
* **RAG answers "could not find it"** – ingestion may have failed (bad key or rate limit at first start). Run `docker compose exec backend python -m app.cli ingest --force`.
* **`API key not valid` in backend logs** – wrong or expired key.
* **Rebuild from scratch** – `docker compose down -v` then `docker compose up --build`.

## Notes and next steps

* Conversation memory is in-process; use `langgraph-checkpoint-postgres` to persist it across restarts.
* The API has no authentication and CORS is open — add auth before exposing it beyond your machine.
* `cancel_order` acts immediately when the user asks; for production consider adding a confirmation step (LangGraph `interrupt`).
* Embeddings use Gemini (`models/gemini-embedding-001`), stored in a separate Chroma collection (`company_policies_gemini`), so an older OpenAI-built index is never mixed in. If you change `EMBEDDING_MODEL`, rebuild with `docker compose exec backend python -m app.cli ingest --force`.
* Chroma server and client are both pinned to 0.5.23 — keep them in sync if you upgrade.

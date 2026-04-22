# multiagent-rag-langgraph

A small AI application demo:
- Supervisor routes user intent to agents: `travel / joke / song / other`
- `song` agent uses Redis-based RAG over couplets corpus for rhyme/structure reference
- `travel` agent calls AMap tools via MCP

## Tech Stack
- Python, asyncio
- LangGraph / LangChain
- DeepSeek Chat (`langchain_deepseek`)
- Zhipu Embedding (via OpenAI-compatible SDK with `base_url`)
- Redis (Docker)

## Project Structure
- `Director.py`: LangGraph multi-agent workflow (supervisor + agents)
- `rag_redis.py`: Redis corpus import + embedding + retrieval
- `data/couplets.csv`: couplets corpus (text1,text2)

## Setup

### 1) Start Redis (Docker)
```bash
docker run -d --name redis -p 6379:6379 redis:latest
```

### 2) Configure environment variables
Create a `.env` file (DO NOT commit):
```env
DEEPSEEK_API_KEY=...
ZHIPU_API_KEY=...
AMAP_MAPS_API_KEY=...   # optional, only required for travel agent

REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Run
```bash
python Director.py
```

## Notes
- RAG corpus import is "lazy once": if Redis already has `couplet:*` keys, importing will be skipped.
- If RAG retrieval fails (quota/network), the song agent falls back to direct generation.

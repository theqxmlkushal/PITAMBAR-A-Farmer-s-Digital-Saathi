## PITAMBAR – Agricultural Advisory Multi‑Agent System  
## 📁 Final File Structure

```
PITAMBAR/
├── app/
│   ├── main.py
│   ├── routes/query_route.py
│   └── controllers/query_controller.py
├── core/
│   ├── orchestrator.py
│   ├── router.py
│   ├── state.py
│   └── synthesizer.py
├── agents/
│   ├── base_agent.py
│   ├── market_agent.py
│   ├── policy_agent.py
│   ├── tech_agent.py
│   └── crop_agent.py
├── tools/
│   ├── rag/
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── apis/
│   │   ├── weather_api.py
│   │   ├── market_api.py
│   │   └── govt_scheme_api.py
│   └── scraping/
│       └── rss_parser.py
├── llm/
│   ├── groq_client.py
│   └── prompts/
│       ├── router_prompt.txt
│       ├── market_prompt.txt
│       ├── policy_prompt.txt
│       ├── tech_prompt.txt
│       ├── crop_prompt.txt
│       └── synthesis_prompt.txt
├── memory/
│   ├── session_memory.py
│   └── user_profile.py
├── multimodal/
│   ├── speech_to_text.py
│   └── text_to_speech.py
├── config/
│   └── settings.py
├── data/documents/          (place your PDF/text files here)
├── tests/
└── requirements.txt
```

---

## 🔧 Setup Instructions

### 1. Clone & environment
```bash
git clone <your-repo>
cd PITAMBAR
python -m venv venv
source venv/bin/activate      # or .\venv\Scripts\activate on Windows
```

### 2. Install dependencies
Create `requirements.txt` with:

```
fastapi
uvicorn
langgraph
groq
qdrant-client
sentence-transformers
duckduckgo-search
feedparser
openai-whisper
gTTS
python-multipart
pydantic
```

Then:
```bash
pip install -r requirements.txt
```

### 3. Environment variables
Create `.env` (or set in `config/settings.py`):
```
GROQ_API_KEY=your_free_groq_key
RSS_FEED_URLS=https://example.com/agri-rss.xml
```

### 4. Run the server
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Test a query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the current price of wheat in Punjab?", "session_id": "test123"}'
```

Voice endpoint: `POST /query/voice` with audio file.

---

## 🧪 Testing Each Component

Every Python file includes a test block. You can run them individually, e.g.:
```bash
python agents/market_agent.py
python tools/rag/vector_store.py
```

---

## 📝 Key Implementation Rules 

1. **No hardcoded strings** – all prompts go into `/llm/prompts/` as `.txt` files.  
2. **Async everywhere** – all agent functions, API calls, and the orchestrator must be `async`.  
3. **State format** must exactly match `core/state.py` (TypedDict).  
4. **Do not add any library** not listed in the spec without asking first.  
5. **Free tier only** – no API that requires payment (OpenAI, Twilio, etc.).  

---

## 🚀 Next Steps (after this chat)

- Implement each module following the structure above.  
- Populate `/data/documents/` with sample agricultural PDFs (e.g., crop guides, scheme brochures).  
- Build the LangGraph state graph in `orchestrator.py`.  
- Integrate session memory and user profiles.  
- Add voice endpoints and test with real farmer queries.

---

## 📄 License & Credits

- Built for Indian farmers – open source, free to use.  
- LLM: Groq (Llama 3.3 70B) via free tier.  
- Weather: Open‑Meteo.  
- Speech: Whisper (OpenAI) & gTTS.
  
---

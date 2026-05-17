# Smart Job Match Agent

Semantic job recommendation API that matches resumes to jobs using Gemini embeddings, cosine similarity ranking, and a two-step LLM agent with real tool calling.

**Live URL:** https://your-project.vercel.app

---

## Tech Stack

- **Framework:** FastAPI (Python)
- **LLM & Embeddings:** Google Gemini API (`gemini-2.5-flash` for generation, `gemini-embedding-001` for embeddings)
- **Vector Math:** NumPy (cosine similarity)
- **Validation:** Pydantic v2 with field validators
- **Deployment:** Vercel (serverless Python)
- **Other:** `python-dotenv`, `uvicorn`

---

## Project Structure

```
Smart Job Match Agent/
├── api/
│   └── index.py          # FastAPI app — endpoints, agent, embedding logic
├── jobs.json             # 50-job dataset (loaded at startup)
├── requirements.txt      # Python dependencies
├── vercel.json           # Vercel routing and build config
├── test_score_spread.py   # Score spread validation test
├── .env.example          # Required environment variables
└── .gitignore
```

---

## How It Works

The API loads 50 jobs from `jobs.json` and embeds them all at startup using Gemini's embedding model. When a resume is submitted to `/recommend`, it is embedded and ranked against the job corpus via cosine similarity. A two-step LLM agent then parses the resume into a structured candidate profile (tool call 1: `parse_resume`) and generates per-job fit/misfit explanations (tool call 2: `reason_about_matches`). A clarifying question is also generated to enable a follow-up `/refine` call that re-ranks jobs based on the candidate's answer.

---

## Setup

```bash
git clone https://github.com/your-username/Smart-Job-Match-Agent.git
cd Smart-Job-Match-Agent
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`):

```
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_API_KEY_2=your_second_gemini_api_key_here
```

Start the server:

```bash
uvicorn api.index:app --reload
```

---

## API Endpoints

### `GET /`

Returns API metadata and available endpoints.

```bash
curl http://localhost:8000/
```

**Response:**

```json
{
  "name": "Smart Job Match API",
  "version": "1.0.0",
  "endpoints": {
    "health": "GET /health",
    "recommend": "POST /recommend",
    "refine": "POST /refine"
  },
  "docs": "/docs"
}
```

---

### `GET /health`

Verifies the API is running, jobs are loaded, and Gemini is reachable.

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "ok",
  "jobs_loaded": 50,
  "embeddings_ready": true,
  "gemini_connected": true,
  "model_embedding": "gemini-embedding-001",
  "model_llm": "gemini-2.5-flash"
}
```

---

### `POST /recommend`

Accepts resume text and returns ranked job recommendations with explanations and a clarifying question.

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Jeet Darji. B.Tech Computer Science student at Charusat University. Skills: Python, Machine Learning, NLP, FastAPI, PyTorch, LangChain, RAG. Built an AI chatbot using LLMs and a resume parser using spaCy. Interested in AI/ML engineering roles. Fresher with internship experience in data science."
  }'
```

**Response shape:**

```json
{
  "candidate": {
    "name": "Jeet Darji",
    "skills": ["Python", "Machine Learning", "NLP", "FastAPI", "PyTorch", "LangChain", "RAG"],
    "experience_years": 0,
    "preferred_roles": ["AI/ML Engineer", "Data Scientist"]
  },
  "ranked_jobs": [
    {
      "id": 13,
      "title": "Generative AI Engineer",
      "company": "GenAI Studio",
      "similarity_score": 0.95,
      "raw_similarity_score": 0.82,
      "explanation": "Strong fit — your LangChain and RAG experience directly maps to this role's requirements..."
    }
  ],
  "clarifying_question": "Do you have a preference for remote roles, or are you open to relocating for on-site positions?",
  "agent_status": "ok"
}
```

`ranked_jobs` contains up to 10 jobs. `agent_status` is `"ok"` or `"degraded"` if the LLM agent failed (results still returned via fallback).

---

### `POST /refine`

Re-ranks jobs after the candidate answers the clarifying question.

```bash
curl -X POST http://localhost:8000/refine \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Jeet Darji. B.Tech Computer Science student at Charusat University. Skills: Python, Machine Learning, NLP, FastAPI, PyTorch, LangChain, RAG. Built an AI chatbot using LLMs and a resume parser using spaCy. Interested in AI/ML engineering roles. Fresher with internship experience in data science.",
    "clarifying_question": "Do you have a preference for remote roles, or are you open to relocating for on-site positions?",
    "candidate_answer": "I strongly prefer remote roles. I want to work from home."
  }'
```

**Response shape:**

```json
{
  "ranked_jobs": [
    {
      "id": 13,
      "title": "Generative AI Engineer",
      "company": "GenAI Studio",
      "similarity_score": 0.95,
      "raw_similarity_score": 0.82,
      "explanation": ""
    }
  ],
  "reasoning": "Boosted remote-friendly roles to the top since the candidate strongly prefers working from home. On-site-only positions were deprioritized."
}
```

---

## Testing

`test_score_spread.py` validates that similarity scores have meaningful separation across the top 10 results. It requires the server to be running locally.

```bash
uvicorn api.index:app --reload   # in one terminal
python test_score_spread.py      # in another terminal
```

**Checks performed:**

1. Scores are not all clustered above 0.90
2. Score range (max − min) across top 10 is ≥ 0.15
3. Top score falls in the 0.80–0.96 range
4. Bottom score falls in the 0.40–0.70 range
5. All scores are between 0.0 and 1.0
6. Scores are in descending order

Exits with code `0` if all checks pass, `1` otherwise.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Primary Google Gemini API key |
| `GEMINI_API_KEY_2` | No | Secondary key for automatic rate-limit rotation |

See `.env.example` for the template.

---

## Vercel Deployment Notes

- **60-second execution timeout** — the two-step agent + embedding pipeline fits within this limit under normal conditions.
- **Cold start on first request** — embedding all 50 jobs takes ~3-5 seconds on a cold start. Subsequent requests are fast.
- **No persistent disk** — `jobs.json` is bundled in the deployment and loaded into memory at startup on every cold start.
- **Two API keys supported** — the `GeminiClientManager` automatically rotates to the second key on rate-limit errors (`429` / `RESOURCE_EXHAUSTED`), reducing failures under load.

---


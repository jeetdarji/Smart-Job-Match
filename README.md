# Smart Job Match Agent

Semantic job recommendation API that matches resumes to jobs using Gemini embeddings, cosine similarity ranking, and a two-step LLM agent with real tool calling.

**Live URL:** https://smart-job-match-lake.vercel.app

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
├── test_score_spread.py  # Score spread validation test
├── README.md             # Project docs + technical write-up (see bottom)
├── .env.example          # Required environment variables
└── .gitignore
```

---

## How It Works

The API loads 50 jobs from `jobs.json` and embeds them all at startup using Gemini's embedding model. When a resume is submitted to `/recommend`, it is embedded and ranked against the job corpus via cosine similarity. A two-step LLM agent then parses the resume into a structured candidate profile (tool call 1: `parse_resume`) and generates per-job fit/misfit explanations (tool call 2: `reason_about_matches`). A clarifying question is also generated to enable a follow-up `/refine` call that re-ranks jobs based on the candidate's answer.

---

## Run Locally (5 commands)

```bash
git clone https://github.com/jeetdarji/Smart-Job-Match.git
cd Smart-Job-Match
pip install -r requirements.txt
cp .env.example .env        # then add your Gemini API key(s)
uvicorn api.index:app --reload
```

> **Note:** You must edit `.env` and add at least `GEMINI_API_KEY` (get one free at [aistudio.google.com](https://aistudio.google.com/)). `GEMINI_API_KEY_2` is optional — used for automatic rate-limit rotation. No real API keys are committed to the repo (`.env` is gitignored).

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
    "preferred_roles": ["AI/ML Engineer", "Data Scientist"],
    "education": "B.Tech Computer Science, Charusat University"
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

## Assignment Context

This project was built as part of the **Cantilever Labs AI Engineer Intern** assignment. The requirements included building a job recommendation API with semantic similarity ranking using embeddings, an LLM agent with real tool calling (not prompt-only), a clarifying question flow, and deployment on Vercel with `POST /recommend` and `POST /refine` endpoints.

---

# Technical Write-up

## 1. Design Choices

I chose **Google Gemini's `gemini-embedding-001`** for embeddings and **`gemini-2.5-flash`** for the LLM agent. The primary reason was Vercel's **1024 MB memory limit** — locally loaded sentence-transformers models (e.g. `all-MiniLM-L6-v2` at ~80 MB, or `all-mpnet-base-v2` at ~420 MB) would consume a significant portion of that budget before any request processing begins. An API-based embedding service avoids this entirely and keeps cold starts under 5 seconds.

I considered **OpenAI `text-embedding-3-small`** as well — it's a strong model with 1536 dimensions and good benchmark scores. I went with Gemini because it let me use a single provider for both embeddings and LLM generation, reducing the number of API keys, SDK dependencies, and failure surfaces. Gemini's embedding model also supports batched embedding in a single API call, which I use at startup to embed all 50 jobs in one request rather than 50 sequential calls.

**Score normalization trade-off:** Raw cosine similarities from high-dimensional embeddings tend to cluster in a narrow band (e.g. 0.65–0.82), which makes it hard to visually distinguish strong matches from weak ones. I apply min-max normalization to spread the top-10 scores into a 0.45–0.95 range so the ranking is visually meaningful. This is a cosmetic transformation — it preserves rank order and does not affect which jobs appear or their relative ordering. The raw cosine similarity is always returned alongside as `raw_similarity_score` for full transparency. I made this choice deliberately: a user seeing scores of 0.78, 0.76, 0.75 gets less signal than seeing 0.92, 0.71, 0.55.

To improve embedding quality, `build_job_corpus_text()` repeats high-signal fields (title, domain, skills) so the embedding captures them more strongly. This is a lightweight form of field boosting without needing a separate retrieval framework.

## 2. Agentic Architecture

The agent runs a **two-step sequential tool-calling loop** using Gemini's native function-calling API:

```
Resume text
    │
    ▼
┌───────────────────────┐
│ Step 1: parse_resume  │  ← Gemini tool call (mode=ANY, forced)
│ Extracts: name,       │
│ skills, experience,   │
│ preferred_roles,      │
│ education             │
└───────────┬───────────┘
            │ structured candidate profile
            ▼
┌─────────────────────────────┐
│ Step 2: reason_about_matches│  ← Gemini tool call (mode=ANY, forced)
│ Input: candidate + top 5    │
│ Output: per-job explanation │
└───────────┬─────────────────┘
            │
            ▼
  Clarifying question (separate LLM call)
```

**Why two tool calls instead of one large prompt?** Separation of concerns — parsing a resume is a fundamentally different task from reasoning about job fit. Combining them would produce a single monolithic prompt where errors in extraction bleed into reasoning. With two calls, I can validate the parsed profile before feeding it to the reasoning step, and I can retry either step independently on failure. It also makes debugging straightforward: if explanations are wrong, I know the issue is in step 2, not the parser.

I use `FunctionCallingConfig(mode=ANY)` to force the model to always produce a tool call rather than a text response. This eliminates a common failure mode where the LLM decides to "helpfully" answer in prose instead of calling the tool.

**Failure modes:** (1) If the LLM returns malformed tool arguments (e.g. `experience_years` as a string), the Pydantic model catches it. (2) Rate limits — I mitigate this with a `GeminiClientManager` that rotates between two API keys automatically on 429 errors. (3) If the agent fails entirely, the `/recommend` endpoint degrades gracefully — it still returns ranked jobs from the embedding layer, just without explanations (`agent_status: "degraded"`).

## 3. Honest Weaknesses

**Noisy resumes:** The system assumes reasonably well-written resume text. A resume that is mostly bullet points with abbreviations ("Py, JS, k8s, tf") or one that is pasted from a PDF with broken formatting (headers mixed into paragraphs, table cells concatenated) would produce a weaker embedding and confuse the resume parser. The embedding model would still produce a vector, but it would be less meaningful.

**Scale (10,000 concurrent requests):** The current design would not survive this. Every `/recommend` call makes 3 synchronous Gemini API calls (embed resume + 2 agent tool calls + 1 clarifying question call). At 10K concurrency, we would instantly hit Gemini rate limits and Vercel's concurrency cap. Fixes would include: embedding caching (pre-compute and store job embeddings in a vector DB like Pinecone), async request handling, request queuing, and horizontal scaling behind a load balancer.

**Corners cut:** (1) Job embeddings are recomputed on every Vercel cold start rather than cached in a persistent store. (2) The `/refine` endpoint re-embeds the resume from scratch instead of caching the embedding from the prior `/recommend` call — adding a session/cache layer would fix this. (3) No unit tests for the agent tool-calling logic; only an integration test (`test_score_spread.py`) for score distribution.

## 4. Next Steps

If I had two more days, the **single highest-impact improvement** would be adding a **vector database (Pinecone or Qdrant)** to pre-store job embeddings and enable fast approximate nearest-neighbor search. This eliminates cold-start re-embedding of all 50 jobs (which takes 3–5 seconds and one large API call), makes the system ready for a much larger job dataset (thousands of jobs), and removes the runtime memory cost of holding all embeddings in a NumPy array. It would also let me cache resume embeddings per session, making the `/refine` endpoint faster by skipping the redundant embedding call.

A close second would be adding **structured evaluation** — a test suite with 5–10 diverse sample resumes and expected top-3 job matches, so I can measure whether changes to the embedding strategy or prompts actually improve recommendation quality rather than relying on manual spot-checking.

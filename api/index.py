import os
import json
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()


class GeminiClientManager:
    """Manages multiple Gemini API keys with automatic rotation on rate limits."""

    def __init__(self):
        self._keys: list[str] = []
        self._clients: list[genai.Client] = []
        self._current_index: int = 0

        for env_var in ["GEMINI_API_KEY", "GEMINI_API_KEY_2"]:
            key = os.getenv(env_var, "").strip()
            if key:
                self._keys.append(key)
                self._clients.append(genai.Client(api_key=key))

        if not self._clients:
            self._keys.append("")
            self._clients.append(genai.Client(api_key=""))

        print(f"[GeminiClientManager] Initialized with {len(self._clients)} API key(s)")

    @property
    def client(self) -> genai.Client:
        return self._clients[self._current_index]

    def rotate(self) -> bool:
        """Switch to the next API key. Returns True if rotation happened."""
        if len(self._clients) <= 1:
            return False
        old_index = self._current_index
        self._current_index = (self._current_index + 1) % len(self._clients)
        print(f"[GeminiClientManager] Rotated key {old_index + 1} → {self._current_index + 1}")
        return True

    def call_with_retry(self, func, *args, **kwargs):
        """Call a function using the current client; on rate limit, rotate and retry."""
        last_error = None
        for attempt in range(len(self._clients)):
            try:
                return func(self.client, *args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = "quota" in error_msg or "resource_exhausted" in error_msg or "rate" in error_msg or "429" in error_msg
                if is_rate_limit and self.rotate():
                    print(f"[GeminiClientManager] Rate limit hit, retrying with key {self._current_index + 1}")
                    last_error = e
                    continue
                raise
        raise last_error


_manager = GeminiClientManager()

# ---------------------------------------------------------------------------
# Load jobs dataset at startup (module level – Vercel compatible)
# ---------------------------------------------------------------------------
JOBS_PATH = Path(__file__).resolve().parent.parent / "jobs.json"

try:
    with open(JOBS_PATH, "r", encoding="utf-8") as f:
        JOBS: list[dict] = json.load(f)
except FileNotFoundError:
    print(f"ERROR: jobs.json not found at {JOBS_PATH}")
    JOBS = []
except json.JSONDecodeError as e:
    print(f"ERROR: jobs.json contains invalid JSON – {e}")
    JOBS = []

# ---------------------------------------------------------------------------
# Phase 2: Embedding helpers
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """Embed a single text using Gemini gemini-embedding-001."""
    try:
        result = _manager.call_with_retry(
            lambda c, t: c.models.embed_content(model="gemini-embedding-001", contents=t),
            text,
        )
        return result.embeddings[0].values
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR: Embedding failed – {e}")
        if "API_KEY" in error_msg or "authentication" in error_msg.lower():
            raise HTTPException(status_code=503, detail="Gemini API key is invalid or missing")
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise HTTPException(status_code=503, detail="Gemini API rate limit hit. Please try again.")
        else:
            raise HTTPException(status_code=503, detail="Embedding service is temporarily unavailable. Please try again later.")


def build_job_corpus_text(job: dict) -> str:
    """Combine job fields into a rich text string for embedding."""
    skills_str = ", ".join(job.get("skills", []))
    remote_str = "Remote work available" if job.get("remote") else "On-site position only"
    domain = job.get("domain", "N/A")
    exp = job.get("experience_years", 0)
    salary = job.get("salary_lpa", "N/A")
    title = job.get("title", "")

    # Repeat domain and title for stronger signal
    domain_signal = f"{domain} {domain} {domain}"
    title_signal = f"{title} {title}"

    return (
        f"{title_signal}. {job.get('company', '')}. "
        f"Industry domain: {domain_signal}. "
        f"Work arrangement: {remote_str}. "
        f"Location: {job.get('location', 'N/A')}. "
        f"Required technical skills: {skills_str}. {skills_str}. "
        f"Years of experience required: {exp} years. "
        f"Salary: {salary} LPA. "
        f"Job description: {job.get('description', '')} "
        f"This is a {domain} role requiring {skills_str}."
    )


# ---------------------------------------------------------------------------
# Embed all jobs at startup (one batched API call)
# ---------------------------------------------------------------------------
JOB_EMBEDDINGS: list[dict] = []

try:
    if JOBS:
        corpus_texts = [build_job_corpus_text(job) for job in JOBS]
        _emb_result = _manager.call_with_retry(
            lambda c, texts: c.models.embed_content(model="gemini-embedding-001", contents=texts),
            corpus_texts,
        )
        for _job, _emb_obj in zip(JOBS, _emb_result.embeddings):
            JOB_EMBEDDINGS.append({
                "job": _job,
                "embedding": np.array(_emb_obj.values),
            })
        print(f"Embedded {len(JOB_EMBEDDINGS)} jobs successfully")
except Exception as e:
    print(f"WARNING: Job embedding failed at startup – {e}")
    JOB_EMBEDDINGS = []


# ---------------------------------------------------------------------------
# Phase 2: Cosine similarity + ranking
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a, vec_b) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return round(float(np.dot(a, b) / (norm_a * norm_b)), 4)


def get_top_n_jobs(resume_embedding: list, n: int = 10) -> list[dict]:
    """Return top N jobs ranked by cosine similarity to resume embedding."""
    resume_vec = np.asarray(resume_embedding, dtype=np.float64)
    scored = []
    for item in JOB_EMBEDDINGS:
        score = cosine_similarity(resume_vec, item["embedding"])
        scored.append({"job": item["job"], "score": score, "raw_score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)

    if not scored:
        return []

    # Apply min-max normalization across ALL scores to spread range
    all_scores = [s["score"] for s in scored]
    min_score = min(all_scores)
    max_score = max(all_scores)
    score_range = max_score - min_score

    if score_range > 0.01:  # Only normalize if there is meaningful spread
        for s in scored:
            # Normalize to 0.45 - 0.95 range to look realistic
            normalized = (s["score"] - min_score) / score_range
            s["score"] = round(0.45 + normalized * 0.50, 4)

    return scored[:n]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Smart Job Match API",
    description="Semantic job matching powered by Gemini embeddings and an LLM agent.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Middleware & Global Error Handling
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    print(f"[REQUEST] {request.method} {request.url.path}")
    response = await call_next(request)
    duration = round(time.time() - start_time, 2)
    print(f"[RESPONSE] {request.method} {request.url.path} → {response.status_code} ({duration}s)")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[UNHANDLED ERROR] {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred. Please try again later."},
    )

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    resume_text: str

    @field_validator("resume_text")
    @classmethod
    def validate_resume_text(cls, v: str) -> str:
        v = v.strip() if v else ""
        if not v:
            raise ValueError("resume_text cannot be empty")
        if len(v) < 50:
            raise ValueError(
                "resume_text is too short to generate meaningful matches. "
                "Please provide a more detailed resume (minimum 50 characters)."
            )
        if len(v) > 50000:
            raise ValueError("resume_text exceeds maximum allowed length")
        return v


class RefineRequest(BaseModel):
    resume_text: str
    clarifying_question: str
    candidate_answer: str

    @field_validator("resume_text")
    @classmethod
    def validate_resume_text(cls, v: str) -> str:
        v = v.strip() if v else ""
        if not v:
            raise ValueError("resume_text cannot be empty")
        if len(v) < 50:
            raise ValueError(
                "resume_text is too short to generate meaningful matches. "
                "Please provide a more detailed resume (minimum 50 characters)."
            )
        if len(v) > 50000:
            raise ValueError("resume_text exceeds maximum allowed length")
        return v

    @field_validator("clarifying_question")
    @classmethod
    def validate_clarifying_question(cls, v: str) -> str:
        v = v.strip() if v else ""
        if not v:
            raise ValueError("clarifying_question cannot be empty")
        if len(v) < 10:
            raise ValueError("clarifying_question is too short (minimum 10 characters)")
        if len(v) > 1000:
            raise ValueError("clarifying_question exceeds maximum allowed length")
        return v

    @field_validator("candidate_answer")
    @classmethod
    def validate_candidate_answer(cls, v: str) -> str:
        v = v.strip() if v else ""
        if not v:
            raise ValueError("candidate_answer cannot be empty")
        if len(v) < 3:
            raise ValueError("candidate_answer is too short (minimum 3 characters)")
        if len(v) > 5000:
            raise ValueError("candidate_answer exceeds maximum allowed length")
        return v


class CandidateProfile(BaseModel):
    name: str
    skills: list[str]
    experience_years: float
    preferred_roles: list[str] = []
    education: str = ""


class RankedJob(BaseModel):
    id: int
    title: str
    company: str
    similarity_score: float
    raw_similarity_score: float = 0.0
    explanation: str


class RecommendResponse(BaseModel):
    candidate: CandidateProfile
    ranked_jobs: list[RankedJob]
    clarifying_question: str
    agent_status: str = "ok"


class RefineResponse(BaseModel):
    ranked_jobs: list[RankedJob]
    reasoning: str

# ---------------------------------------------------------------------------
# Phase 3: Tool definitions for Gemini function calling
# ---------------------------------------------------------------------------

TOOL_PARSE_RESUME = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="parse_resume",
            description="Extract structured information from raw resume text. Call this first to understand the candidate before matching jobs.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "name": types.Schema(
                        type=types.Type.STRING,
                        description="Full name of the candidate"
                    ),
                    "skills": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="List of technical and soft skills mentioned in the resume"
                    ),
                    "experience_years": types.Schema(
                        type=types.Type.NUMBER,
                        description="Total years of professional experience. Use 0 for freshers or students."
                    ),
                    "preferred_roles": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="Job roles or titles the candidate seems to target"
                    ),
                    "education": types.Schema(
                        type=types.Type.STRING,
                        description="Highest education qualification and field of study"
                    ),
                    "domain_interests": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="Industry domains the candidate has experience or interest in"
                    )
                },
                required=["name", "skills", "experience_years", "preferred_roles", "education"]
            )
        )
    ]
)

TOOL_REASON_MATCHES = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="reason_about_matches",
            description="Given a candidate profile and their top job matches, generate natural language explanations for each job explaining fit or misfit.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "explanations": types.Schema(
                        type=types.Type.ARRAY,
                        description="List of explanations one per job in the same order as input jobs",
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "job_id": types.Schema(
                                    type=types.Type.INTEGER,
                                    description="The id field of the job"
                                ),
                                "explanation": types.Schema(
                                    type=types.Type.STRING,
                                    description="2-3 sentences explaining fit or misfit for this specific candidate. Be specific about which skills match, which are missing, and why the domain aligns or does not."
                                )
                            },
                            required=["job_id", "explanation"]
                        )
                    )
                },
                required=["explanations"]
            )
        )
    ]
)

AGENT_SYSTEM_PROMPT = (
    "You are an expert technical recruiter. Your job is to carefully analyze "
    "resumes and job matches. Always use the tools provided — never respond "
    "with plain text."
)


def run_agent(resume_text: str, top_jobs: list[dict]) -> tuple[CandidateProfile, dict]:
    """Two-step agentic loop: parse resume then reason about matches."""

    try:
        system_instruction = AGENT_SYSTEM_PROMPT

        # --- Step 1: Parse resume ---
        print("[run_agent] Step 1: Parsing resume with tool call...")

        step1_response = _manager.call_with_retry(
            lambda c, content, cfg: c.models.generate_content(model="gemini-2.5-flash", contents=content, config=cfg),
            f"Parse this resume and extract structured information:\n\n{resume_text}",
            types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[TOOL_PARSE_RESUME],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=["parse_resume"]
                    )
                )
            )
        )

        step1_part = step1_response.candidates[0].content.parts[0]

        if not hasattr(step1_part, 'function_call') or step1_part.function_call is None:
            raise ValueError("Model did not return a tool call for parse_resume")

        parsed_args = dict(step1_part.function_call.args)
        print(f"[run_agent] Resume parsed: name={parsed_args.get('name')}, skills_count={len(parsed_args.get('skills', []))}")

        candidate = CandidateProfile(
            name=parsed_args.get("name", "Unknown"),
            skills=parsed_args.get("skills", []),
            experience_years=float(parsed_args.get("experience_years", 0)),
            preferred_roles=parsed_args.get("preferred_roles", []),
            education=parsed_args.get("education", ""),
        )

        # --- Step 2: Reason about matches ---
        print("[run_agent] Step 2: Reasoning about job matches with tool call...")

        job_context = "\n".join([
            f"Job ID {j['job']['id']}: {j['job']['title']} at {j['job']['company']} "
            f"(Domain: {j['job']['domain']}, "
            f"Skills: {', '.join(j['job']['skills'][:4])}, "
            f"Remote: {j['job']['remote']})"
            for j in top_jobs
        ])

        candidate_summary = (
            f"Candidate: {candidate.name}\n"
            f"Skills: {', '.join(candidate.skills)}\n"
            f"Experience: {candidate.experience_years} years\n"
            f"Preferred roles: {', '.join(parsed_args.get('preferred_roles', []))}"
        )

        step2_response = _manager.call_with_retry(
            lambda c, content, cfg: c.models.generate_content(model="gemini-2.5-flash", contents=content, config=cfg),
            f"{candidate_summary}\n\nTop 5 job matches:\n{job_context}\n\nGenerate fit/misfit explanations for each job for this specific candidate.",
            types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[TOOL_REASON_MATCHES],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=["reason_about_matches"]
                    )
                )
            )
        )

        step2_part = step2_response.candidates[0].content.parts[0]

        if not hasattr(step2_part, 'function_call') or step2_part.function_call is None:
            raise ValueError("Model did not return a tool call for reason_about_matches")

        explanations_raw = list(step2_part.function_call.args.get("explanations", []))
        explanations = {
            int(dict(e)["job_id"]): dict(e)["explanation"]
            for e in explanations_raw
        }
        print(f"[run_agent] Explanations generated for {len(explanations)} jobs")

        return candidate, explanations

    except Exception as e:
        print(f"[run_agent] Agent failed: {str(e)}")
        raise


def generate_clarifying_question(
    resume_text: str, candidate: CandidateProfile, top_jobs: list[dict]
) -> str:
    """Generate one smart clarifying question via a direct LLM call."""
    domains = list({item["job"].get("domain", "") for item in top_jobs})
    titles = [item["job"]["title"] for item in top_jobs[:5]]
    remote_count = sum(1 for item in top_jobs[:5] if item["job"].get("remote", False))

    user_message = (
        f"Candidate skills: {', '.join(candidate.skills)}\n"
        f"Experience: {candidate.experience_years} years\n"
        f"Top matched job domains: {', '.join(domains)}\n"
        f"Top matched job titles: {', '.join(titles)}\n"
        f"Remote-friendly jobs in top 5: {remote_count} out of 5\n\n"
        f"Generate exactly one smart clarifying question."
    )

    try:
        response = _manager.call_with_retry(
            lambda c, content, cfg: c.models.generate_content(model="gemini-2.5-flash", contents=content, config=cfg),
            user_message,
            types.GenerateContentConfig(
                system_instruction=(
                    "You are a smart technical recruiter assistant. "
                    "Based on the candidate profile and their top job matches, "
                    "generate exactly ONE specific insightful follow-up question "
                    "that resolves a real ambiguity or gap you noticed. "
                    "Do not ask generic questions like 'tell me more about yourself'. "
                    "Be specific. No preamble. No numbering. Just the question."
                ),
                max_output_tokens=300,
                temperature=0.7
            )
        )

        question = response.text.strip()
        print(f"[generate_clarifying_question] Generated: {question}")
        return question

    except Exception as e:
        print(f"[generate_clarifying_question] Failed: {e}")
        return "Could you tell me more about the types of roles or domains you are most interested in?"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_jobs_for_refine(ranked_items: list[dict]) -> str:
    """Format ranked jobs as a numbered list for the refine LLM prompt."""
    lines = []
    for i, item in enumerate(ranked_items, 1):
        job = item["job"]
        top_skills = ", ".join(job.get("skills", [])[:3])
        remote_str = "Remote" if job.get("remote") else "On-site"
        lines.append(
            f"{i}. [ID:{job['id']}] {job['title']} at {job['company']} "
            f"(Domain: {job.get('domain')}, {remote_str}, Skills: {top_skills})"
        )
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "Smart Job Match API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "recommend": "POST /recommend",
            "refine": "POST /refine"
        },
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Verify the API is running, jobs are loaded, and Gemini is reachable."""
    gemini_connected = False
    try:
        test_result = _manager.call_with_retry(
            lambda c, t: c.models.embed_content(model="gemini-embedding-001", contents=t),
            "health check"
        )
        gemini_connected = True
    except Exception:
        gemini_connected = False

    return {
        "status": "ok",
        "jobs_loaded": len(JOBS),
        "embeddings_ready": len(JOB_EMBEDDINGS) > 0,
        "gemini_connected": gemini_connected,
        "model_embedding": "gemini-embedding-001",
        "model_llm": "gemini-2.5-flash",
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """Return ranked job recommendations for a given resume."""
    # Step 1: Input validation is handled by Pydantic field_validator

    # Step 2: Embed resume
    print(f"[/recommend] Embedding resume ({len(req.resume_text)} chars)")
    resume_embedding = embed_text(req.resume_text)
    print(f"[/recommend] Resume embedded successfully. Vector length: {len(resume_embedding)}")

    # Step 3: Rank jobs
    all_ranked = get_top_n_jobs(resume_embedding, n=10)
    if not all_ranked:
        raise HTTPException(
            status_code=500,
            detail="Job ranking failed — no embeddings available",
        )
    print(f"[/recommend] Top job: {all_ranked[0]['job']['title']} score={all_ranked[0]['score']}")
    print(f"[/recommend] Score range: {all_ranked[-1]['score']} to {all_ranked[0]['score']}")
    print(f"[/recommend] Score spread: min={all_ranked[-1]['score']} max={all_ranked[0]['score']} range={round(all_ranked[0]['score']-all_ranked[-1]['score'],4)}")
    top_5 = all_ranked[:5]

    # Step 4: Agent (graceful degradation)
    agent_status = "ok"
    try:
        candidate, explanations = run_agent(req.resume_text, top_5)
        print(f"[/recommend] Agent succeeded: {candidate.name}, {len(candidate.skills)} skills")
    except Exception as e:
        print(f"[/recommend] Agent failed, using fallback: {e}")
        candidate = CandidateProfile(name="Unknown", skills=[], experience_years=0.0, education="")
        explanations = []
        agent_status = "degraded"

    # Step 5: Clarifying question (graceful degradation)
    try:
        question = generate_clarifying_question(req.resume_text, candidate, top_5)
    except Exception as e:
        print(f"[/recommend] Clarifying question failed: {e}")
        question = (
            "Could you tell me more about the types of roles or "
            "domains you are most interested in?"
        )

    # Step 6: Response assembly
    explanation_map = explanations if isinstance(explanations, dict) else {exp["job_id"]: exp["explanation"] for exp in explanations}
    ranked_jobs = []
    for item in all_ranked:
        job = item["job"]
        ranked_jobs.append(RankedJob(
            id=job["id"],
            title=job["title"],
            company=job["company"],
            similarity_score=item["score"],
            raw_similarity_score=item.get("raw_score", item["score"]),
            explanation=explanation_map.get(job["id"], ""),
        ))

    return RecommendResponse(
        candidate=candidate,
        ranked_jobs=ranked_jobs,
        clarifying_question=question,
        agent_status=agent_status,
    )


@app.post("/refine", response_model=RefineResponse)
def refine(req: RefineRequest):
    """Re-rank jobs after the candidate answers a clarifying question."""
    # Step 1: Input validation is handled by Pydantic field_validator

    # Step 2: Re-embed and base rank
    print(f"[/refine] Embedding resume ({len(req.resume_text)} chars)")
    resume_embedding = embed_text(req.resume_text)
    base_ranked = get_top_n_jobs(resume_embedding, n=10)
    if not base_ranked:
        raise HTTPException(
            status_code=500,
            detail="Job ranking failed — no embeddings available",
        )

    # Step 3: LLM re-ranking call
    jobs_context = format_jobs_for_refine(base_ranked)
    item_by_id = {item["job"]["id"]: item for item in base_ranked}
    raw = ""

    user_message = (
        f"Original resume summary:\n{req.resume_text[:500]}\n\n"
        f"Clarifying question that was asked:\n{req.clarifying_question}\n\n"
        f"Candidate's answer:\n{req.candidate_answer}\n\n"
        f"Current top 10 job matches (in current ranked order):\n{jobs_context}\n\n"
        f"Re-rank these jobs based on the candidate's answer. Consider:\n"
        f"- If they mentioned preference for remote/onsite → reprioritize accordingly\n"
        f"- If they mentioned a domain preference → boost jobs in that domain\n"
        f"- If they revealed a skill → boost jobs requiring that skill\n"
        f"- If they revealed a constraint (location, salary) → deprioritize conflicting jobs\n\n"
        f"Return ONLY the JSON object."
    )

    try:
        refine_response = _manager.call_with_retry(
            lambda c, content, cfg: c.models.generate_content(model="gemini-2.5-flash", contents=content, config=cfg),
            user_message,
            types.GenerateContentConfig(
                system_instruction=(
                    "You are a job matching expert. Re-rank jobs based on new candidate information. "
                    "Respond with valid JSON only. "
                    "Format: {\"new_order\": [list of job IDs best first], "
                    "\"reasoning\": \"2-3 sentences explaining what changed and why\"}"
                ),
                response_mime_type="application/json",
                max_output_tokens=8192,
                temperature=0.3
            )
        )

        raw = refine_response.text.strip()
        # Strip markdown code fences
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
        # Extract JSON object if there is surrounding text
        if not raw.startswith("{"):
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]
        refine_data = json.loads(raw)
        new_order_raw = refine_data.get("new_order", refine_data.get("ranked_ids", []))
        new_order = [int(jid) for jid in new_order_raw]
        reasoning = refine_data.get("reasoning", "")
        print(f"[/refine] LLM returned new order: {new_order}")
    except json.JSONDecodeError as e:
        print(f"WARNING: Refine JSON parsing failed – {e}. Raw: {raw[:200]}")
        new_order = [item["job"]["id"] for item in base_ranked]
        reasoning = "Could not re-rank. Returning original semantic ranking."
    except Exception as e:
        error_msg = str(e)
        print(f"WARNING: Refine LLM call failed – {e}")
        if "API_KEY" in error_msg or "authentication" in error_msg.lower():
            raise HTTPException(status_code=503, detail="Gemini API key is invalid or missing")
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            raise HTTPException(status_code=503, detail="Gemini API rate limit hit. Please try again.")
        else:
            new_order = [item["job"]["id"] for item in base_ranked]
            reasoning = "Could not re-rank. Returning original semantic ranking."

    # Step 4: Reorder and respond
    # Scores preserved from semantic ranking; order changed by LLM reasoning
    ranked_jobs: list[RankedJob] = []
    seen: set[int] = set()
    for jid in new_order:
        if jid in item_by_id and jid not in seen:
            seen.add(jid)
            item = item_by_id[jid]
            job = item["job"]
            ranked_jobs.append(RankedJob(
                id=job["id"],
                title=job["title"],
                company=job["company"],
                similarity_score=item["score"],
                raw_similarity_score=item.get("raw_score", item["score"]),
                explanation="",
            ))
    # Append any jobs missed by the LLM
    for item in base_ranked:
        jid = item["job"]["id"]
        if jid not in seen:
            seen.add(jid)
            job = item["job"]
            ranked_jobs.append(RankedJob(
                id=job["id"],
                title=job["title"],
                company=job["company"],
                similarity_score=item["score"],
                raw_similarity_score=item.get("raw_score", item["score"]),
                explanation="",
            ))

    return RefineResponse(ranked_jobs=ranked_jobs, reasoning=reasoning)

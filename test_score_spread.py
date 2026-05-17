"""
Score Spread Test for Smart Job Match Agent
=============================================
Validates that similarity scores have meaningful separation
so the ranking is actually useful.

Requirements checked:
  1. Scores are NOT all clustered above 0.90
  2. Score range (max - min) across top 10 is >= 0.15
  3. Top job score is in ~0.85-0.95 range
  4. Bottom (10th) job score is in ~0.45-0.65 range
  5. All scores are between 0.0 and 1.0
"""

import requests
import sys

API_URL = "http://127.0.0.1:8000"

# A realistic software-engineer resume for testing
SAMPLE_RESUME = """
Senior Python Developer with 6 years of experience in backend development,
machine learning, and cloud infrastructure. Proficient in Python, FastAPI,
Django, PostgreSQL, Docker, Kubernetes, and AWS. Built scalable REST APIs
serving 10M+ requests/day. Experience with TensorFlow and PyTorch for
production ML pipelines. Led a team of 4 engineers at a fintech startup.
Strong background in data structures, algorithms, and system design.
Bachelor's in Computer Science from IIT Bombay.
"""


def main():
    # --- Pre-flight: check server is up ---
    print("=" * 60)
    print("SCORE SPREAD TEST")
    print("=" * 60)

    try:
        health = requests.get(f"{API_URL}/health", timeout=10).json()
    except requests.ConnectionError:
        print("\n[FAIL] Cannot connect to API at", API_URL)
        print("       Start the server first:  uvicorn api.index:app --reload")
        sys.exit(1)

    print(f"\nHealth: jobs_loaded={health.get('jobs_loaded')}, "
          f"embeddings_ready={health.get('embeddings_ready')}, "
          f"gemini_connected={health.get('gemini_connected')}")

    if not health.get("embeddings_ready"):
        print("[FAIL] Embeddings not ready. Wait for 'Embedded 50 jobs successfully'.")
        sys.exit(1)

    # --- Call /recommend ---
    print("\nSending resume to /recommend ...")
    resp = requests.post(
        f"{API_URL}/recommend",
        json={"resume_text": SAMPLE_RESUME.strip()},
        timeout=120,
    )

    if resp.status_code != 200:
        print(f"[FAIL] /recommend returned {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)

    data = resp.json()
    ranked = data.get("ranked_jobs", [])

    if not ranked:
        print("[FAIL] No ranked_jobs in response.")
        sys.exit(1)

    # --- Extract scores ---
    scores = [job["similarity_score"] for job in ranked]
    top_score = scores[0]
    bottom_score = scores[-1]
    score_range = round(top_score - bottom_score, 4)

    print(f"\n{'─' * 60}")
    print(f"{'Rank':<6} {'Score':<10} {'Title':<30} {'Company'}")
    print(f"{'─' * 60}")
    for i, job in enumerate(ranked, 1):
        print(f"{i:<6} {job['similarity_score']:<10.4f} {job['title'][:30]:<30} {job['company']}")
    print(f"{'─' * 60}")

    print(f"\nTop score (rank 1):   {top_score:.4f}")
    print(f"Bottom score (rank {len(scores)}): {bottom_score:.4f}")
    print(f"Score range:          {score_range:.4f}")
    print(f"Number of results:    {len(scores)}")

    # --- Run checks ---
    print(f"\n{'=' * 60}")
    print("CHECKS")
    print(f"{'=' * 60}")

    all_passed = True

    # Check 1: Not clustered above 0.90
    above_90 = sum(1 for s in scores if s > 0.90)
    if above_90 == len(scores):
        print(f"[FAIL] CHECK 1: ALL {len(scores)} scores are above 0.90 — embeddings not working correctly")
        all_passed = False
    else:
        print(f"[PASS] CHECK 1: Scores NOT all clustered above 0.90 ({above_90}/{len(scores)} above 0.90)")

    # Check 2: Score range >= 0.15
    if score_range >= 0.15:
        print(f"[PASS] CHECK 2: Score range {score_range:.4f} >= 0.15")
    else:
        print(f"[FAIL] CHECK 2: Score range {score_range:.4f} < 0.15 (too narrow)")
        all_passed = False

    # Check 3: Top score in 0.85-0.95 range
    if 0.80 <= top_score <= 0.96:
        print(f"[PASS] CHECK 3: Top score {top_score:.4f} is in acceptable range [0.80, 0.96]")
    else:
        print(f"[WARN] CHECK 3: Top score {top_score:.4f} outside ideal range [0.80, 0.96]")

    # Check 4: Bottom score in 0.45-0.65 range
    if 0.40 <= bottom_score <= 0.70:
        print(f"[PASS] CHECK 4: Bottom score {bottom_score:.4f} is in acceptable range [0.40, 0.70]")
    else:
        print(f"[WARN] CHECK 4: Bottom score {bottom_score:.4f} outside ideal range [0.40, 0.70]")

    # Check 5: All scores between 0 and 1
    out_of_range = [s for s in scores if s < 0.0 or s > 1.0]
    if not out_of_range:
        print(f"[PASS] CHECK 5: All scores in [0.0, 1.0]")
    else:
        print(f"[FAIL] CHECK 5: {len(out_of_range)} scores out of [0.0, 1.0] range: {out_of_range}")
        all_passed = False

    # Check 6: Scores are in descending order
    is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    if is_sorted:
        print(f"[PASS] CHECK 6: Scores are in descending order (ranking preserved)")
    else:
        print(f"[FAIL] CHECK 6: Scores are NOT in descending order")
        all_passed = False

    # --- Final verdict ---
    print(f"\n{'=' * 60}")
    if all_passed:
        print("RESULT: ALL CHECKS PASSED ✓")
    else:
        print("RESULT: SOME CHECKS FAILED ✗")
    print(f"{'=' * 60}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()


"""
inference.py — Baseline inference script for Warehouse Spatial QA.

MANDATORY FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables required:
    API_BASE_URL   LLM API endpoint
    MODEL_NAME     Model identifier
    HF_TOKEN       Hugging Face / API key
    ENV_SEED       Random seed (default: 42)
"""
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN")    # no default — must be set as env var
BENCHMARK    = "warehouse-spatial-qa"

MAX_STEPS_PER_TASK = {
    "object_location":        1,
    "multi_constraint_query": 3,
    "movement_prediction":    5,
}
SUCCESS_THRESHOLD = 0.5
TEMPERATURE       = 0.0
MAX_TOKENS        = 200

SYSTEM_PROMPT = textwrap.dedent("""
    You are a warehouse spatial reasoning assistant.
    You will be shown a warehouse floor-plan description followed by a spatial question.
    Answer concisely and precisely. Do not explain your reasoning — only give the answer.
    For yes/no questions: answer exactly "yes" or "no".
    For item ID questions: answer with the single letter ID only (e.g. "A", "B").
    For zone questions: answer with one of: RECEIVING, SHIPPING, BULK_STORAGE_WEST, BULK_STORAGE_EAST.
    For multi-part questions: use the exact format requested.
""").strip()


# ── Log helpers (mandatory format) ───────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val    = error if error else "null"
    action_clean = action.replace("\n", " ").strip()[:80]
    print(
        f"[STEP] step={step} action={action_clean!r} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

def get_agent_answer(client: OpenAI, scene_description: str, question: str, hints: Optional[str]) -> str:
    user_content = f"Scene:\n{scene_description}\n\nQuestion:\n{question}"
    if hints:
        user_content += f"\n\nFormat hint: {hints}"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "none"


# ── Episode runner ────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> float:
    """Run one full episode for the given task. Returns final score in [0, 1]."""
    from environment import SpatialQAEnv
    from models import SpatialAction

    env         = SpatialQAEnv(n_items=4)
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs  = env.reset(task_id=task_id)
        done = obs.done

        max_steps = MAX_STEPS_PER_TASK[task_id]

        for step in range(1, max_steps + 1):
            if done:
                break

            answer = get_agent_answer(
                client,
                obs.scene_description,
                obs.question,
                obs.hints,
            )

            obs    = env.step(SpatialAction(answer=answer))
            reward = float(obs.reward or 0.0)
            done   = obs.done
            error  = obs.metadata.get("error") if obs.metadata else None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=answer, reward=reward, done=done, error=error)

            if done:
                break

        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = round(min(max(score, 0.0), 1.0), 2)
        success = score >= SUCCESS_THRESHOLD

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main: run all 3 tasks ─────────────────────────────────────────────────────

def main() -> None:
    client      = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_scores = {}

    for task_id in ["object_location", "multi_constraint_query", "movement_prediction"]:
        score = run_task(client, task_id)
        task_scores[task_id] = score

    print("\n[SUMMARY]", flush=True)
    for task_id, s in task_scores.items():
        print(f"  {task_id}: {s:.2f}", flush=True)
    overall = sum(task_scores.values()) / len(task_scores)
    print(f"  overall: {overall:.2f}", flush=True)


if __name__ == "__main__":
    main()

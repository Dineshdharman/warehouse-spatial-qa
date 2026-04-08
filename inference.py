"""
inference.py — RL-style inference script for Warehouse Spatial QA.

The LLM receives its reward and feedback after every step and maintains a
conversation history so it can self-correct on subsequent steps — exactly
like a policy-improvement loop in RL.

MANDATORY FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables required:
    API_BASE_URL   LLM API endpoint
    MODEL_NAME     Model identifier
    HF_TOKEN       Hugging Face / API key (no default)
    ENV_SEED       Random seed (default: 42)
"""
import os
import textwrap
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")    # no default — must be set as env var
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
    When you receive feedback about a wrong answer, carefully re-read the question and correct yourself.
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

def call_llm(client: OpenAI, messages: List[Dict]) -> str:
    """Call the LLM with full conversation history. Returns the answer string."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "none"


# ── RL episode runner ─────────────────────────────────────────────────────────

def run_task_rl(
    client: OpenAI,
    task_id: str,
    episode_num: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one episode with RL-style feedback loop.

    After each step the environment's reward + feedback message is appended
    to the conversation history so the LLM can self-correct on the next step.

    Returns a structured dict with all step data for reporting.
    """
    from environment import SpatialQAEnv
    from models import SpatialAction

    env         = SpatialQAEnv(n_items=4)
    rewards: List[float] = []
    steps_data: List[Dict] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    if verbose:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs  = env.reset(task_id=task_id)
        done = obs.done

        # Build conversation history — persists across all steps in this episode
        messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # First user message: full scene + first question
        first_msg = f"Scene:\n{obs.scene_description}\n\nQuestion:\n{obs.question}"
        if obs.hints:
            first_msg += f"\n\nFormat hint: {obs.hints}"
        messages.append({"role": "user", "content": first_msg})

        max_steps = MAX_STEPS_PER_TASK[task_id]

        for step in range(1, max_steps + 1):
            if done:
                break

            current_question = obs.question

            # ── LLM answers with full conversation history ──────────────────
            answer = call_llm(client, messages)
            messages.append({"role": "assistant", "content": answer})

            # ── Step in environment ─────────────────────────────────────────
            obs    = env.step(SpatialAction(answer=answer))
            reward = float(obs.reward or 0.0)
            done   = obs.done
            feedback = obs.feedback or ""
            error    = obs.metadata.get("error") if obs.metadata else None

            rewards.append(reward)
            steps_taken = step

            if verbose:
                log_step(step=step, action=answer, reward=reward, done=done, error=error)

            steps_data.append({
                "step":     step,
                "question": current_question,
                "answer":   answer,
                "reward":   reward,
                "feedback": feedback,
                "done":     done,
            })

            # ── Feed reward + feedback back into conversation ───────────────
            if not done and feedback:
                next_msg = (
                    f"[RL Feedback] Reward: {reward:.2f}. {feedback}\n\n"
                    f"Next question:\n{obs.question}"
                )
                if obs.hints:
                    next_msg += f"\n\nFormat hint: {obs.hints}"
                messages.append({"role": "user", "content": next_msg})

        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = round(min(max(score, 0.0), 1.0), 2)
        success = score >= SUCCESS_THRESHOLD

    finally:
        env.close()
        if verbose:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "episode":  episode_num + 1,
        "task_id":  task_id,
        "score":    score,
        "success":  success,
        "steps":    steps_data,
        "rewards":  rewards,
    }


# ── Main: run all 3 tasks ─────────────────────────────────────────────────────

def main() -> None:
    client      = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    task_scores = {}

    for task_id in ["object_location", "multi_constraint_query", "movement_prediction"]:
        result          = run_task_rl(client, task_id, episode_num=0, verbose=True)
        task_scores[task_id] = result["score"]

    print("\n[SUMMARY]", flush=True)
    for task_id, s in task_scores.items():
        print(f"  {task_id}: {s:.2f}", flush=True)
    overall = sum(task_scores.values()) / len(task_scores)
    print(f"  overall: {overall:.2f}", flush=True)


if __name__ == "__main__":
    main()

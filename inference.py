"""
Inference Script — DarkStore Pricing Environment
=================================================
Runs an LLM agent against all 3 tasks and emits structured stdout logs.

MANDATORY FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables required:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — model identifier
  HF_TOKEN      — Hugging Face / API key
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ─────────────────────────────── Config ──────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "darkstore-pricing-env"

TASK_MAX_STEPS = {
    "easy_pricing": 7,
    "medium_pricing": 14,
    "hard_pricing": 30,
}

SUCCESS_THRESHOLD = 0.35
TEMPERATURE = 0.5

COST_PRICES = {"milk": 33.0, "banana": 15.0, "bread": 17.0}

# ─────────────────────────────── Logging ─────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────── Prompts ─────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert pricing agent for a dark store (rapid delivery grocery) selling perishable goods.
Your goal is to set optimal prices each day to maximize profit while minimizing spoilage and stockouts.

Products and their cost prices:
- Milk: ₹33/unit (expires in 1 day, restocked daily)
- Banana: ₹15/unit (expires in 2 days, restocked every 2 days)
- Bread: ₹17/unit (expires in 3 days, restocked every 2 days)

Market: 1000 households. Baseline daily demand: Milk=800, Banana=300, Bread=400.
Demand is price-elastic: higher prices reduce demand, lower prices increase demand.
- Milk elasticity: -0.8 (moderate)
- Banana elasticity: -1.4 (elastic — price-sensitive)
- Bread elasticity: -1.1 (moderately elastic)

Pricing strategy hints:
- Price above cost to make profit, but too high kills demand
- Watch expiry dates: reduce prices if items expire soon (move inventory)
- Watch restock schedule: if restock is due tomorrow, use up current stock today
- Stockouts lose revenue too — don't overprice when demand is high
- Optimal margin is typically 20-40% above cost price

You MUST respond with ONLY a valid JSON object in exactly this format:
{
  "milk_price": <number>,
  "banana_price": <number>,
  "bread_price": <number>,
  "reasoning": "<brief 1-2 sentence explanation>"
}
No markdown, no extra text, just the JSON object.
""").strip()


def build_user_prompt(obs: Dict[str, Any], step: int, last_reward: float, history: List[str]) -> str:
    inv = obs.get("inventory", {})
    expiry = obs.get("expiry_units", {})
    forecast = obs.get("demand_forecast", {})
    restock = obs.get("restock_schedule", {})
    hist_sales = obs.get("historical_sales", [])
    hist_prices = obs.get("historical_prices", [])

    history_block = "\n".join(history[-5:]) if history else "None yet"

    exp_str = {
        p: f"exp1d={expiry.get(p,[0,0,0])[0]} exp2d={expiry.get(p,[0,0,0])[1]} exp3d={expiry.get(p,[0,0,0])[2]}"
        for p in ["milk", "banana", "bread"]
    }

    last_sales_str = ""
    if hist_sales:
        last = hist_sales[-1]
        last_prices_entry = hist_prices[-1] if hist_prices else {}
        last_sales_str = (
            f"  Yesterday: Milk sold={last.get('milk',0)} @₹{last_prices_entry.get('milk','?')}, "
            f"Banana sold={last.get('banana',0)} @₹{last_prices_entry.get('banana','?')}, "
            f"Bread sold={last.get('bread',0)} @₹{last_prices_entry.get('bread','?')}"
        )
    else:
        last_sales_str = "  No history yet (Day 1)"

    return textwrap.dedent(f"""
Day {obs.get('day', step)} | Days remaining: {obs.get('days_remaining', '?')} | Task: {obs.get('task_name', '?')}
Last step reward: {last_reward:.3f}

CURRENT INVENTORY:
  Milk:   {inv.get('milk', '?')} units  | {exp_str['milk']} | Restock in {restock.get('milk', '?')} day(s)
  Banana: {inv.get('banana', '?')} units | {exp_str['banana']} | Restock in {restock.get('banana', '?')} day(s)
  Bread:  {inv.get('bread', '?')} units  | {exp_str['bread']} | Restock in {restock.get('bread', '?')} day(s)

DEMAND FORECAST (at cost price):
  Milk: ~{forecast.get('milk', '?')} | Banana: ~{forecast.get('banana', '?')} | Bread: ~{forecast.get('bread', '?')}

RECENT SALES HISTORY:
{last_sales_str}

STEP HISTORY:
{history_block}

Set today's prices to maximize profit. Consider expiry urgency, inventory levels, and demand elasticity.
Respond ONLY with the JSON object as specified.
""").strip()


# ─────────────────────────────── Agent ───────────────────────────────────────

def get_agent_prices(
    client: OpenAI,
    obs: Dict[str, Any],
    step: int,
    last_reward: float,
    history: List[str],
) -> Dict[str, float]:
    """Query the LLM and parse pricing decision."""
    prompt = build_user_prompt(obs, step, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=200,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        return {
            "milk_price": float(parsed.get("milk_price", 40.0)),
            "banana_price": float(parsed.get("banana_price", 20.0)),
            "bread_price": float(parsed.get("bread_price", 22.0)),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as exc:
        print(f"[DEBUG] LLM parse error: {exc}", flush=True)
        # Fallback: price slightly above cost
        return {
            "milk_price": COST_PRICES["milk"] * 1.25,
            "banana_price": COST_PRICES["banana"] * 1.30,
            "bread_price": COST_PRICES["bread"] * 1.28,
            "reasoning": "fallback pricing",
        }


# ─────────────────────────── Environment Client ──────────────────────────────

class EnvClient:
    """HTTP client for the DarkStore environment server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30)

    def reset(self, task_name: str, seed: Optional[int] = None) -> Dict[str, Any]:
        payload = {"task_name": task_name}
        if seed is not None:
            payload["seed"] = seed
        r = self._client.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        return r.json()

    def step(self, task_name: str, prices: Dict[str, float]) -> Dict[str, Any]:
        payload = {
            "task_name": task_name,
            "milk_price": prices["milk_price"],
            "banana_price": prices["banana_price"],
            "bread_price": prices["bread_price"],
        }
        r = self._client.post(f"{self.base_url}/step", json=payload)
        r.raise_for_status()
        return r.json()

    def grade(self, task_name: str) -> Dict[str, Any]:
        r = self._client.get(f"{self.base_url}/grade", params={"task_name": task_name})
        r.raise_for_status()
        return r.json()

    def close(self):
        self._client.close()


# ─────────────────────────────── Main ────────────────────────────────────────

def run_task(client: OpenAI, env_client: EnvClient, task_name: str, seed: int = 42) -> float:
    max_steps = TASK_MAX_STEPS[task_name]
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_data = env_client.reset(task_name=task_name, seed=seed)
        obs = reset_data["observation"]
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            # Get pricing decision from LLM
            prices = get_agent_prices(client, obs, step, last_reward, history)

            action_str = (
                f"milk={prices['milk_price']:.1f},"
                f"banana={prices['banana_price']:.1f},"
                f"bread={prices['bread_price']:.1f}"
            )

            try:
                step_data = env_client.step(task_name=task_name, prices=prices)
                obs = step_data["observation"]
                reward = float(step_data.get("reward", 0.0))
                done = bool(step_data.get("done", False))
                info = step_data.get("info", {})
                last_error = None

                rewards.append(reward)
                steps_taken = step
                last_reward = reward

                log_step(step=step, action=action_str, reward=reward, done=done, error=None)

                # Record history for prompt
                profit = info.get("profit", 0)
                sales = info.get("sales", {})
                history.append(
                    f"Day {obs.get('day',step)}: {action_str} -> "
                    f"profit=₹{profit:.0f} sales={sales.get('milk',0)}/{sales.get('banana',0)}/{sales.get('bread',0)} "
                    f"reward={reward:.3f}"
                )

                if done:
                    break

            except Exception as e:
                last_error = str(e)
                log_step(step=step, action=action_str, reward=0.0, done=False, error=last_error)
                steps_taken = step
                rewards.append(0.0)
                break

        # Get final grade
        try:
            grade_data = env_client.grade(task_name=task_name)
            score = float(grade_data.get("score", 0.0))
        except Exception:
            score = sum(rewards) / len(rewards) if rewards else 0.0

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        last_error = str(e)
        print(f"[DEBUG] Task {task_name} failed: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = EnvClient(base_url=ENV_BASE_URL)

    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] ENV_BASE_URL={ENV_BASE_URL}", flush=True)

    all_scores = {}
    for task_name in ["easy_pricing", "medium_pricing", "hard_pricing"]:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)
        score = run_task(client, env_client, task_name, seed=42)
        all_scores[task_name] = score
        time.sleep(1)

    print(f"\n{'='*60}", flush=True)
    print("FINAL SCORES:", flush=True)
    for t, s in all_scores.items():
        print(f"  {t}: {s:.4f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"  AVERAGE: {avg:.4f}", flush=True)

    env_client.close()


if __name__ == "__main__":
    main()

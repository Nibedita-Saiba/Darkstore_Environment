"""
FastAPI server exposing OpenEnv-compatible HTTP endpoints for the DarkStore environment.
Also serves a web visualization dashboard when ENABLE_WEB_INTERFACE=true.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from darkstore_env import (
    DarkStoreAction,
    DarkStoreEnv,
    DarkStoreObservation,
    GRADERS,
    PRODUCTS,
    COST_PRICE,
)

# ─────────────────────────── App Setup ────────────────────────────────────────

app = FastAPI(
    title="DarkStore Pricing Environment",
    description="OpenEnv-compatible perishable goods pricing simulation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ENABLE_WEB = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() == "true"

# Global environment registry (task_name → env instance)
_envs: Dict[str, DarkStoreEnv] = {}


def get_env(task_name: str) -> DarkStoreEnv:
    if task_name not in _envs:
        _envs[task_name] = DarkStoreEnv(task_name=task_name)
    return _envs[task_name]


# ─────────────────────────── Request Models ──────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = "medium_pricing"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    task_name: Optional[str] = "medium_pricing"
    milk_price: float = 40.0
    banana_price: float = 20.0
    bread_price: float = 22.0


# ─────────────────────────── Endpoints ───────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "darkstore-pricing-env", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    if req is None:
        req = ResetRequest()
    task_name = req.task_name or "medium_pricing"
    env = DarkStoreEnv(task_name=task_name, seed=req.seed)
    _envs[task_name] = env
    obs = env.reset()
    return {
        "observation": obs.dict(),
        "task_name": task_name,
        "message": f"Environment reset for task '{task_name}'",
    }


@app.post("/step")
def step(req: StepRequest):
    task_name = req.task_name or "medium_pricing"
    if task_name not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"Environment for task '{task_name}' not initialized. Call /reset first.",
        )
    env = _envs[task_name]
    action = DarkStoreAction(
        milk_price=req.milk_price,
        banana_price=req.banana_price,
        bread_price=req.bread_price,
    )
    result = env.step(action)
    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
def state(task_name: str = "medium_pricing"):
    if task_name not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"Environment for task '{task_name}' not initialized.",
        )
    return _envs[task_name].state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "easy_pricing",
                "description": "Price Milk only for 7 days",
                "difficulty": "easy",
                "days": 7,
                "active_products": ["milk"],
            },
            {
                "name": "medium_pricing",
                "description": "Price all 3 products for 14 days with restocking",
                "difficulty": "medium",
                "days": 14,
                "active_products": ["milk", "banana", "bread"],
            },
            {
                "name": "hard_pricing",
                "description": "30-day pricing with demand shocks and spoilage",
                "difficulty": "hard",
                "days": 30,
                "active_products": ["milk", "banana", "bread"],
            },
        ]
    }


@app.post("/grade")
def grade(task_name: str = "medium_pricing"):
    if task_name not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"No environment to grade for task '{task_name}'.",
        )
    env = _envs[task_name]
    grader = GRADERS.get(task_name)
    if not grader:
        raise HTTPException(status_code=400, detail=f"No grader for task '{task_name}'.")
    score = grader.grade(env)
    return {
        "task_name": task_name,
        "score": score,
        "cumulative_profit": env.cumulative_profit,
        "days_completed": env.day,
        "episode_done": env.done,
    }


# ─────────────────────────── Web Dashboard ───────────────────────────────────

WEB_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DarkStore Pricing Environment</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  :root {
    --bg: #0d0f14;
    --surface: #161a22;
    --border: #252b38;
    --accent: #00e5a0;
    --accent2: #ff6b35;
    --accent3: #7b8cff;
    --text: #e8ecf0;
    --muted: #6b7685;
    --danger: #ff4757;
    --warn: #ffd32a;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    min-height: 100vh;
    padding: 24px;
  }

  header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: -0.5px;
  }

  .badge {
    background: var(--accent);
    color: #000;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 3px;
  }

  h2 {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--muted);
    margin-bottom: 12px;
  }

  .grid {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 20px;
    align-items: start;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
  }

  .task-selector {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 16px;
  }

  .task-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 10px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    text-align: left;
    transition: all 0.15s;
  }

  .task-btn:hover { border-color: var(--accent); color: var(--accent); }
  .task-btn.active { border-color: var(--accent); background: rgba(0,229,160,0.07); color: var(--accent); }

  .price-input-group {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 16px;
  }

  .price-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .price-label {
    width: 80px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: var(--muted);
  }

  .price-input {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
    padding: 7px 10px;
    outline: none;
    transition: border-color 0.15s;
  }

  .price-input:focus { border-color: var(--accent); }

  .cost-hint {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    white-space: nowrap;
  }

  .btn {
    width: 100%;
    padding: 11px;
    border-radius: 6px;
    border: none;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    margin-bottom: 8px;
  }

  .btn-primary { background: var(--accent); color: #000; }
  .btn-primary:hover { background: #00c88d; }
  .btn-secondary { background: transparent; border: 1px solid var(--border); color: var(--text); }
  .btn-secondary:hover { border-color: var(--accent2); color: var(--accent2); }

  .metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }

  .metric-row:last-child { border-bottom: none; }
  .metric-val { font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
  .positive { color: var(--accent); }
  .negative { color: var(--danger); }
  .neutral { color: var(--text); }
  .warning { color: var(--warn); }

  .inventory-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin-bottom: 16px;
  }

  .inv-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    text-align: center;
  }

  .inv-product {
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
  }

  .inv-qty {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
  }

  .inv-expiry {
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
  }

  .log-container {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    height: 320px;
    overflow-y: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    line-height: 1.6;
  }

  .log-line { margin-bottom: 4px; }
  .log-day { color: var(--accent3); }
  .log-profit { }
  .log-spoil { color: var(--danger); }
  .log-reward { color: var(--accent); }

  .progress-bar-container {
    background: var(--bg);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin-top: 8px;
  }

  .progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent3));
    border-radius: 4px;
    transition: width 0.4s ease;
  }

  .score-display {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 40px;
    font-weight: 600;
    color: var(--accent);
    text-align: center;
    padding: 16px 0;
  }

  .status-indicator {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    margin-right: 6px;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .shock-badge {
    background: var(--danger);
    color: #fff;
    font-size: 10px;
    font-family: 'IBM Plex Mono', monospace;
    padding: 2px 6px;
    border-radius: 3px;
    margin-left: 6px;
  }

  #status-msg {
    font-size: 13px;
    color: var(--muted);
    margin-top: 6px;
    min-height: 18px;
  }

  .right-col > .card { margin-bottom: 16px; }
</style>
</head>
<body>
<header>
  <div class="logo">DARKSTORE_ENV</div>
  <div class="badge">OpenEnv</div>
  <span style="color: var(--muted); font-size: 13px; margin-left: auto;">
    <span class="status-indicator"></span>Live Simulation
  </span>
</header>

<div class="grid">
  <!-- LEFT PANEL -->
  <div class="left-col">
    <div class="card">
      <h2>Select Task</h2>
      <div class="task-selector">
        <button class="task-btn" onclick="selectTask('easy_pricing')">
          🟢 Easy — Milk Only (7 days)
        </button>
        <button class="task-btn active" onclick="selectTask('medium_pricing')">
          🟡 Medium — All Products (14 days)
        </button>
        <button class="task-btn" onclick="selectTask('hard_pricing')">
          🔴 Hard — Demand Shocks (30 days)
        </button>
      </div>
      <button class="btn btn-secondary" onclick="resetEnv()">↺ Reset Episode</button>
    </div>

    <div class="card">
      <h2>Set Prices (₹)</h2>
      <div class="price-input-group">
        <div class="price-row">
          <span class="price-label">🥛 Milk</span>
          <input class="price-input" id="milk_price" type="number" value="42" min="1" step="0.5">
          <span class="cost-hint">cost ₹33</span>
        </div>
        <div class="price-row">
          <span class="price-label">🍌 Banana</span>
          <input class="price-input" id="banana_price" type="number" value="22" min="1" step="0.5">
          <span class="cost-hint">cost ₹15</span>
        </div>
        <div class="price-row">
          <span class="price-label">🍞 Bread</span>
          <input class="price-input" id="bread_price" type="number" value="24" min="1" step="0.5">
          <span class="cost-hint">cost ₹17</span>
        </div>
      </div>
      <button class="btn btn-primary" onclick="doStep()">→ Submit Prices / Next Day</button>
      <div id="status-msg">Ready — reset to begin</div>
    </div>

    <div class="card">
      <h2>Episode Score</h2>
      <div class="score-display" id="score-display">—</div>
      <div style="font-size:12px; color: var(--muted); text-align:center;">avg normalized reward</div>
      <div class="progress-bar-container">
        <div class="progress-bar" id="progress-bar" style="width:0%"></div>
      </div>
      <div style="font-size:12px; color: var(--muted); text-align:center; margin-top:6px;" id="day-label">Day 0</div>
    </div>
  </div>

  <!-- RIGHT PANEL -->
  <div class="right-col">
    <div class="card">
      <h2>Inventory</h2>
      <div class="inventory-grid">
        <div class="inv-card">
          <div class="inv-product">🥛 Milk</div>
          <div class="inv-qty neutral" id="inv-milk">—</div>
          <div class="inv-expiry" id="exp-milk">—</div>
        </div>
        <div class="inv-card">
          <div class="inv-product">🍌 Banana</div>
          <div class="inv-qty neutral" id="inv-banana">—</div>
          <div class="inv-expiry" id="exp-banana">—</div>
        </div>
        <div class="inv-card">
          <div class="inv-product">🍞 Bread</div>
          <div class="inv-qty neutral" id="inv-bread">—</div>
          <div class="inv-expiry" id="exp-bread">—</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Today's Results</h2>
      <div id="results-panel">
        <div style="color: var(--muted); font-size:13px;">No data yet — submit prices to begin.</div>
      </div>
    </div>

    <div class="card">
      <h2>Activity Log</h2>
      <div class="log-container" id="log-container">
        <div class="log-line" style="color: var(--muted)">// Environment ready. Select a task and reset to begin.</div>
      </div>
    </div>
  </div>
</div>

<script>
  let currentTask = 'medium_pricing';
  let totalDays = 14;
  let currentDay = 0;
  let stepRewards = [];

  function selectTask(name) {
    currentTask = name;
    const days = {easy_pricing: 7, medium_pricing: 14, hard_pricing: 30};
    totalDays = days[name];
    document.querySelectorAll('.task-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    setStatus(`Task selected: ${name}. Click Reset to begin.`, 'neutral');
  }

  async function resetEnv() {
    const res = await fetch('/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task_name: currentTask})
    });
    const data = await res.json();
    currentDay = 0;
    stepRewards = [];
    updateInventory(data.observation);
    updateScore(0);
    updateDayLabel(0);
    clearResults();
    addLog(`// Episode reset — Task: ${currentTask}`, 'log-day');
    setStatus('Episode reset. Set prices and submit.', 'neutral');
  }

  async function doStep() {
    const milk = parseFloat(document.getElementById('milk_price').value);
    const banana = parseFloat(document.getElementById('banana_price').value);
    const bread = parseFloat(document.getElementById('bread_price').value);

    setStatus('Processing...', 'neutral');

    const res = await fetch('/step', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        task_name: currentTask,
        milk_price: milk,
        banana_price: banana,
        bread_price: bread
      })
    });

    if (!res.ok) {
      const err = await res.json();
      setStatus(`Error: ${err.detail}`, 'negative');
      return;
    }

    const data = await res.json();
    const obs = data.observation;
    const info = data.info;
    const reward = data.reward;
    const done = data.done;

    currentDay = obs.day;
    stepRewards.push(reward);

    updateInventory(obs);
    updateResults(info, reward, done);
    updateDayLabel(currentDay);

    const avgReward = stepRewards.reduce((a,b) => a+b, 0) / stepRewards.length;
    updateScore(avgReward);

    // Log line
    const profit = info.profit;
    const expired = Object.values(info.expired_units).reduce((a,b)=>a+b,0);
    const shocks = Object.keys(info.demand_shocks || {});
    let logLine = `Day ${currentDay}: P₹${milk}/${banana}/${bread} | Profit: ₹${profit.toFixed(0)} | Reward: ${reward.toFixed(3)}`;
    if (expired > 0) logLine += ` | ⚠ Expired: ${expired}`;
    if (shocks.length > 0) logLine += ` | ⚡Shock: ${shocks.join(',')}`;

    addLog(logLine, profit >= 0 ? 'log-profit' : 'log-spoil');

    if (done) {
      const gradeRes = await fetch(`/grade?task_name=${currentTask}`);
      const gradeData = await gradeRes.json();
      addLog(`// Episode complete! Final score: ${gradeData.score.toFixed(4)}`, 'log-reward');
      setStatus(`Episode complete! Score: ${gradeData.score.toFixed(4)}`, 'positive');
    } else {
      setStatus(`Day ${currentDay} complete. ${obs.days_remaining} days remaining.`, 'neutral');
    }
  }

  function updateInventory(obs) {
    for (const p of ['milk', 'banana', 'bread']) {
      const qty = obs.inventory[p];
      const el = document.getElementById(`inv-${p}`);
      el.textContent = qty;
      el.className = `inv-qty ${qty < 100 ? 'warning' : qty === 0 ? 'negative' : 'positive'}`;

      const exp = obs.expiry_units[p];
      const expEl = document.getElementById(`exp-${p}`);
      expEl.textContent = `exp1d:${exp[0]} exp2d:${exp[1]} exp3d:${exp[2]}`;
    }
  }

  function updateResults(info, reward, done) {
    const panel = document.getElementById('results-panel');
    const sales = info.sales;
    const expired = info.expired_units;
    const profit = info.profit;
    const breakdown = info.reward_breakdown;

    panel.innerHTML = `
      <div class="metric-row">
        <span>Daily Profit</span>
        <span class="metric-val ${profit >= 0 ? 'positive' : 'negative'}">₹${profit.toFixed(2)}</span>
      </div>
      <div class="metric-row">
        <span>Step Reward</span>
        <span class="metric-val ${reward > 0.5 ? 'positive' : reward > 0.3 ? 'warning' : 'negative'}">${reward.toFixed(4)}</span>
      </div>
      <div class="metric-row">
        <span>Sales (Milk / Banana / Bread)</span>
        <span class="metric-val neutral">${sales.milk} / ${sales.banana} / ${sales.bread}</span>
      </div>
      <div class="metric-row">
        <span>Expired Units</span>
        <span class="metric-val ${expired.milk+expired.banana+expired.bread > 0 ? 'negative' : 'positive'}">${expired.milk} / ${expired.banana} / ${expired.bread}</span>
      </div>
      <div class="metric-row">
        <span>Spoilage Penalty</span>
        <span class="metric-val negative">₹${breakdown.spoilage_penalty.toFixed(2)}</span>
      </div>
      <div class="metric-row">
        <span>Stockout Penalty</span>
        <span class="metric-val negative">₹${breakdown.stockout_penalty.toFixed(2)}</span>
      </div>
      <div class="metric-row">
        <span>Demand Shocks</span>
        <span class="metric-val warning">${Object.keys(info.demand_shocks||{}).join(', ') || 'none'}</span>
      </div>
    `;
  }

  function clearResults() {
    document.getElementById('results-panel').innerHTML =
      '<div style="color: var(--muted); font-size:13px;">No data yet.</div>';
  }

  function updateScore(val) {
    document.getElementById('score-display').textContent = val.toFixed(4);
    document.getElementById('progress-bar').style.width = `${Math.min(val * 100, 100)}%`;
  }

  function updateDayLabel(day) {
    document.getElementById('day-label').textContent = `Day ${day} / ${totalDays}`;
  }

  function addLog(msg, cls = '') {
    const container = document.getElementById('log-container');
    const div = document.createElement('div');
    div.className = `log-line ${cls}`;
    div.textContent = msg;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }

  function setStatus(msg, cls = '') {
    const el = document.getElementById('status-msg');
    el.textContent = msg;
    el.className = cls;
  }

  // Auto-reset on load
  resetEnv();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if ENABLE_WEB:
        return HTMLResponse(content=WEB_DASHBOARD_HTML)
    return HTMLResponse(content="<h2>DarkStore Pricing Env — API Running</h2><p>Set ENABLE_WEB_INTERFACE=true for dashboard.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

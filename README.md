# 🏪 DarkStore Pricing Environment

An **OpenEnv-compatible** simulation of a perishable goods dark store (rapid delivery grocery), where an AI agent sets daily prices for Milk, Banana, and Bread to maximize profit while managing inventory expiry, restocking schedules, and real demand elasticity.

---

## 🎯 Environment Description & Motivation

Dark stores operate under extreme margin pressure: items expire fast, demand is price-elastic, and restocking is rigid. This environment captures exactly these constraints — making it a rich, real-world task where optimal pricing requires:

- Understanding **price elasticity** (higher price → lower demand, not linearly)
- Anticipating **expiry dates** (markdown strategy before spoilage)
- Reading **restock schedules** (don't overstock perishable items)
- Avoiding **stockouts** (lost revenue) and **wastage** (sunk cost)

The environment is non-trivial: the same price on Day 1 yields a different reward on Day 3 due to stochastic demand noise, changing inventory levels, and restock events. This forces agents to reason over trajectories, not just single steps.

---

## 📦 Products

| Product | Cost Price | Shelf Life | Restock Frequency | Baseline Demand |
|---------|-----------|-----------|-------------------|----------------|
| 🥛 Milk | ₹33/unit | 1 day (24h) | Daily | 800 units/day |
| 🍌 Banana | ₹15/unit | 2 days (36h) | Every 2 days | 300 units/day |
| 🍞 Bread | ₹17/unit | 3 days (60h) | Every 2 days | 400 units/day |

**Price elasticities:** Milk = -0.8 (inelastic), Banana = -1.4 (elastic), Bread = -1.1 (moderate)

---

## 🔭 Observation Space

```json
{
  "day": 5,
  "inventory": {"milk": 620, "banana": 180, "bread": 310},
  "expiry_units": {
    "milk": [620, 0, 0],
    "banana": [120, 60, 0],
    "bread": [0, 150, 160]
  },
  "cost_prices": {"milk": 33.0, "banana": 15.0, "bread": 17.0},
  "historical_sales": [...],
  "historical_prices": [...],
  "demand_forecast": {"milk": 792.4, "banana": 304.1, "bread": 396.8},
  "task_name": "medium_pricing",
  "days_remaining": 9,
  "restock_schedule": {"milk": 1, "banana": 0, "bread": 1}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `day` | int | Current simulation day |
| `inventory` | dict | Units in stock per product |
| `expiry_units` | dict | Units expiring in 1/2/3 days |
| `cost_prices` | dict | Wholesale cost per unit |
| `historical_sales` | list | Last 5 days of sales data |
| `historical_prices` | list | Last 5 days of price decisions |
| `demand_forecast` | dict | Noisy forecast of baseline demand |
| `days_remaining` | int | Days left in the episode |
| `restock_schedule` | dict | Days until next restock per product |

---

## 🎬 Action Space

```json
{
  "milk_price": 42.0,
  "banana_price": 22.0,
  "bread_price": 24.0
}
```

All prices must be ≥ ₹1. Prices below cost trigger heavy penalties.

---

## 🏆 Tasks

### Task 1: `easy_pricing` (Easy) — 7 days
- **Goal:** Price **Milk only** for 7 days
- **Products active:** Milk only
- **Demand shocks:** None
- **Expected score:** 0.55–0.75 for a naive agent, 0.80+ for optimal

Banana and bread prices are still accepted but only milk affects scoring. Focus on understanding milk's daily restocking and 24h expiry.

### Task 2: `medium_pricing` (Medium) — 14 days
- **Goal:** Price **all 3 products** over 14 days
- **Products active:** Milk, Banana, Bread
- **Demand shocks:** None
- **Expected score:** 0.40–0.60 naive, 0.70+ optimal

Agent must juggle different restocking intervals and expiry windows. Banana and bread accumulate across days — markdown before expiry is critical.

### Task 3: `hard_pricing` (Hard) — 30 days
- **Goal:** All 3 products for 30 days with **random demand shocks**
- **Products active:** Milk, Banana, Bread
- **Demand shocks:** 15% chance per product per day (multiplier 0.4–1.8×)
- **Expected score:** 0.30–0.50 naive, 0.65+ optimal

Shocks are visible in the `info` dict. Agent must adapt pricing dynamically in response to revealed shock events across the 30-day horizon.

---

## 💰 Reward Function

Each step returns a reward in **[0, 1]** computed as:

```
total_raw = profit + spoilage_penalty + stockout_penalty + below_cost_penalty
reward = normalize(total_raw)  # clamped to [0, 1]
```

| Component | Formula | Rationale |
|-----------|---------|-----------|
| **Profit** | `(price - cost) × units_sold` | Core objective |
| **Spoilage penalty** | `-0.5 × cost × expired_units` | Penalizes waste |
| **Stockout penalty** | `-0.3 × margin × unmet_demand` | Penalizes lost opportunity |
| **Below-cost penalty** | `-2.0 × loss × units_sold` | Hard disincentive |

### Demand Model

```
Q = Q_base × (P / P_base)^elasticity × lognormal_noise × shock_multiplier
```

Noise is log-normal (μ=0, σ=0.08) — same price on different days yields different demand, preventing agents from exploiting fixed-price strategies.

---

## 🚀 Setup & Usage

### Prerequisites

```bash
python --version   # 3.10+
docker --version
pip install openenv-core fastapi uvicorn httpx openai pydantic
```

### Local Development

```bash
git clone <your-repo>
cd darkstore_env

# Install dependencies
pip install -r requirements.txt

# Start the environment server
python server.py
# → http://localhost:7860 (dashboard if ENABLE_WEB_INTERFACE=true)
```

### Docker

```bash
docker build -t darkstore-env .
docker run -p 7860:7860 -e ENABLE_WEB_INTERFACE=true darkstore-env
```

### Run Inference

```bash
export HF_TOKEN="your-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

### OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset episode `{"task_name": "medium_pricing", "seed": 42}` |
| `/step` | POST | Take action `{"task_name": ..., "milk_price": 42, ...}` |
| `/state` | GET | Full internal state `?task_name=medium_pricing` |
| `/grade` | GET | Compute episode score `?task_name=medium_pricing` |
| `/tasks` | GET | List all tasks |

---

## 📊 Baseline Scores

Baseline agent: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router

| Task | Score |
|------|-------|
| easy_pricing | ~0.62 |
| medium_pricing | ~0.51 |
| hard_pricing | ~0.44 |

---

## 🗂 Project Structure

```
darkstore_env/
├── darkstore_env.py    # Core environment + typed models + graders
├── server.py           # FastAPI server (OpenEnv HTTP endpoints + web UI)
├── inference.py        # Baseline LLM inference script
├── openenv.yaml        # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧠 Training Tips for Agents

1. **Don't price below cost** — the penalty is 2× the loss, never worth it
2. **Reduce banana/bread prices before expiry** — moving inventory at low margin beats wasting it
3. **Milk needs daily attention** — it expires every 24h, always check `expiry_units.milk[0]`
4. **Monitor restock_schedule** — avoid pricing too low right before a large restock arrives
5. **Demand shocks (hard task)** — visible in `info.demand_shocks`; react by adjusting next-day prices

---

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace / API key | — |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:7860` |
| `ENABLE_WEB_INTERFACE` | Enable dashboard UI | `true` |

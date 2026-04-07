"""
DarkStore Pricing Environment — OpenEnv-compatible simulation of a perishable goods
online dark store. Agent sets daily prices; environment calculates demand, sales,
spoilage, and restocking using realistic economic formulas.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ─────────────────────────────── Typed Models ──────────────────────────────────

class DarkStoreObservation(BaseModel):
    day: int = Field(..., description="Current simulation day (1-indexed)")
    inventory: Dict[str, int] = Field(..., description="Current units in stock per product")
    expiry_units: Dict[str, List[int]] = Field(
        ..., description="Units expiring in [1 day, 2 days, 3 days] per product"
    )
    cost_prices: Dict[str, float] = Field(..., description="Wholesale cost per unit")
    historical_sales: List[Dict[str, Any]] = Field(
        default_factory=list, description="Past days' sales data (last 5 days)"
    )
    historical_prices: List[Dict[str, Any]] = Field(
        default_factory=list, description="Past days' price decisions (last 5 days)"
    )
    demand_forecast: Dict[str, float] = Field(
        ..., description="Expected baseline demand at cost-price (noisy estimate)"
    )
    task_name: str = Field(..., description="Active task identifier")
    days_remaining: int = Field(..., description="Days left in the episode")
    restock_schedule: Dict[str, int] = Field(
        ..., description="Days until next restock for each product"
    )


class DarkStoreAction(BaseModel):
    milk_price: float = Field(..., ge=1.0, description="Price per unit of Milk (₹)")
    banana_price: float = Field(..., ge=1.0, description="Price per unit of Banana (₹)")
    bread_price: float = Field(..., ge=1.0, description="Price per unit of Bread (₹)")


class DarkStoreReward(BaseModel):
    total_reward: float = Field(..., description="Normalized reward in [0, 1] for scoring")
    raw_profit: float = Field(..., description="Raw profit/loss in ₹ for the day")
    spoilage_penalty: float = Field(..., description="Penalty for expired units")
    stockout_penalty: float = Field(..., description="Penalty for unmet demand")
    components: Dict[str, float] = Field(..., description="Per-product breakdown")


class DarkStoreStepResult(BaseModel):
    observation: DarkStoreObservation
    reward: float
    done: bool
    info: Dict[str, Any]


# ──────────────────────────── Product Configuration ────────────────────────────

PRODUCTS = ["milk", "banana", "bread"]

COST_PRICE = {"milk": 33.0, "banana": 15.0, "bread": 17.0}

# Shelf life in hours → converted to days (ceiling)
SHELF_LIFE_DAYS = {
    "milk": 1,    # 24 hours
    "banana": 2,  # 36 hours → expires end of day 2
    "bread": 3,   # 60 hours → expires end of day 3
}

# Restock frequency: every N days
RESTOCK_INTERVAL = {"milk": 1, "banana": 2, "bread": 2}

# Normal daily demand at baseline (cost price ≈ typical market price)
BASELINE_DEMAND = {"milk": 800, "banana": 300, "bread": 400}

# Price elasticity of demand (negative: higher price → lower demand)
ELASTICITY = {"milk": -0.8, "banana": -1.4, "bread": -1.1}

# Base restock quantity (slightly above normal demand)
RESTOCK_QTY = {"milk": 850, "banana": 350, "bread": 450}

# Demand shock probabilities (hard task)
DEMAND_SHOCK_PROB = 0.15


# ──────────────────────────────── Core Engine ─────────────────────────────────

class DarkStoreEnv:
    """
    OpenEnv-compatible dark store pricing simulation.

    Tasks:
        easy_pricing   — price Milk only, 7 days
        medium_pricing — price all 3 products, 14 days
        hard_pricing   — all products, 30 days, with demand shocks & stricter scoring
    """

    TASK_CONFIG = {
        "easy_pricing": {
            "days": 7,
            "active_products": ["milk"],
            "demand_shocks": False,
            "description": "Optimize Milk pricing over 7 days",
        },
        "medium_pricing": {
            "days": 14,
            "active_products": ["milk", "banana", "bread"],
            "demand_shocks": False,
            "description": "Optimize all product pricing over 14 days with restocking",
        },
        "hard_pricing": {
            "days": 30,
            "active_products": ["milk", "banana", "bread"],
            "demand_shocks": True,
            "description": "30-day pricing with demand shocks, strict spoilage penalties",
        },
    }

    def __init__(self, task_name: str = "medium_pricing", seed: Optional[int] = None):
        if task_name not in self.TASK_CONFIG:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(self.TASK_CONFIG)}")
        self.task_name = task_name
        self.config = self.TASK_CONFIG[task_name]
        self.active_products = self.config["active_products"]
        self.max_days = self.config["days"]
        self.seed = seed
        self._rng = random.Random(seed)

        # Episode state
        self.day: int = 0
        self.inventory: Dict[str, int] = {}
        # inventory_batches[product] = list of (units, expiry_day) tuples
        self.inventory_batches: Dict[str, List[Tuple[int, int]]] = {}
        self.restock_next: Dict[str, int] = {}
        self.history_sales: List[Dict[str, Any]] = []
        self.history_prices: List[Dict[str, Any]] = []
        self.cumulative_profit: float = 0.0
        self.done: bool = False

        # Tracking for scoring
        self._step_rewards: List[float] = []
        self._demand_shock_active: Dict[str, float] = {}  # product -> multiplier

        # Max theoretical daily profit for normalization
        self._max_daily_profit = sum(
            BASELINE_DEMAND[p] * (COST_PRICE[p] * 0.6)  # assume 60% margin max useful
            for p in self.active_products
        )

    # ──────────────────── OpenEnv Interface ────────────────────

    def reset(self) -> DarkStoreObservation:
        """Reset the environment and return the initial observation."""
        self._rng = random.Random(self.seed)
        self.day = 0
        self.inventory_batches = {p: [] for p in PRODUCTS}
        self.restock_next = {p: 1 for p in PRODUCTS}
        self.history_sales = []
        self.history_prices = []
        self.cumulative_profit = 0.0
        self.done = False
        self._step_rewards = []
        self._demand_shock_active = {}

        # Initial stock: full restock for all products
        for product in PRODUCTS:
            qty = RESTOCK_QTY[product]
            expiry_day = self.day + SHELF_LIFE_DAYS[product]
            self.inventory_batches[product] = [(qty, expiry_day)]

        self._update_inventory_totals()
        return self._make_observation()

    def step(self, action: DarkStoreAction) -> DarkStoreStepResult:
        """
        Apply pricing action, simulate one day of trading, return result.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self.day += 1
        prices = self._extract_prices(action)

        # 1. Expire stale inventory (batches whose expiry_day < today)
        expired = self._expire_inventory()

        # 2. Update demand shocks (hard task)
        self._update_demand_shocks()

        # 3. Calculate actual demand with elasticity + noise
        actual_demand = self._calc_demand(prices)

        # 4. Fulfill sales from inventory (FIFO — oldest first)
        sales, stockout = self._fulfill_sales(actual_demand)

        # 5. Calculate profit
        profit, per_product = self._calc_profit(sales, prices)

        # 6. Restock products due today
        restocked = self._do_restock()

        # 7. Update totals
        self._update_inventory_totals()

        # 8. Calculate reward
        reward_obj = self._calc_reward(profit, expired, stockout, sales, prices)
        raw_reward = reward_obj.total_reward
        self._step_rewards.append(raw_reward)
        self.cumulative_profit += profit

        # 9. Record history
        day_sales = {p: sales.get(p, 0) for p in PRODUCTS}
        day_prices = {p: prices.get(p, COST_PRICE[p]) for p in PRODUCTS}
        self.history_sales.append({"day": self.day, **day_sales})
        self.history_prices.append({"day": self.day, **day_prices})

        done = self.day >= self.max_days
        self.done = done

        info = {
            "day": self.day,
            "profit": profit,
            "per_product_profit": per_product,
            "expired_units": expired,
            "stockout_units": stockout,
            "actual_demand": actual_demand,
            "sales": sales,
            "restocked": restocked,
            "reward_breakdown": reward_obj.dict(),
            "cumulative_profit": self.cumulative_profit,
            "demand_shocks": dict(self._demand_shock_active),
            "noise_applied": getattr(self, "_last_noise", {}),
        }

        obs = self._make_observation()
        return DarkStoreStepResult(
            observation=obs,
            reward=raw_reward,
            done=done,
            info=info,
        )

    def state(self) -> Dict[str, Any]:
        """Return full internal state (for debugging/logging)."""
        return {
            "task_name": self.task_name,
            "day": self.day,
            "max_days": self.max_days,
            "done": self.done,
            "inventory": dict(self.inventory),
            "inventory_batches": {
                p: [(qty, exp) for qty, exp in batches]
                for p, batches in self.inventory_batches.items()
            },
            "restock_next": dict(self.restock_next),
            "cumulative_profit": self.cumulative_profit,
            "step_rewards": list(self._step_rewards),
            "history_sales": list(self.history_sales),
            "history_prices": list(self.history_prices),
            "demand_shocks": dict(self._demand_shock_active),
        }

    def compute_episode_score(self) -> float:
        """
        Compute final normalized score in [0, 1] for the completed episode.
        Used by the grader.
        """
        if not self._step_rewards:
            return 0.0
        avg_reward = sum(self._step_rewards) / len(self._step_rewards)
        # avg_reward is already in [0,1] (clamped per step)
        return round(min(max(avg_reward, 0.0), 1.0), 4)

    # ──────────────────── Internal Helpers ────────────────────

    def _extract_prices(self, action: DarkStoreAction) -> Dict[str, float]:
        prices = {}
        for p in PRODUCTS:
            attr = f"{p}_price"
            prices[p] = getattr(action, attr, COST_PRICE[p])
        return prices

    def _expire_inventory(self) -> Dict[str, int]:
        expired = {p: 0 for p in PRODUCTS}
        for product in PRODUCTS:
            surviving = []
            for qty, expiry_day in self.inventory_batches[product]:
                if expiry_day < self.day:
                    expired[product] += qty
                else:
                    surviving.append((qty, expiry_day))
            self.inventory_batches[product] = surviving
        return expired

    def _update_demand_shocks(self):
        """Random demand shocks for hard task."""
        if not self.config["demand_shocks"]:
            self._demand_shock_active = {}
            return
        for p in PRODUCTS:
            if self._rng.random() < DEMAND_SHOCK_PROB:
                # Shock: demand multiplier between 0.4 (crash) and 1.8 (surge)
                self._demand_shock_active[p] = self._rng.uniform(0.4, 1.8)
            elif p in self._demand_shock_active:
                del self._demand_shock_active[p]

    def _calc_demand(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Demand model using price elasticity:
            Q = Q_base * (P / P_base)^elasticity * noise * shock
        where P_base = cost_price (reference price)
        """
        demand = {}
        noise_log = {}
        for product in PRODUCTS:
            q_base = BASELINE_DEMAND[product]
            p = prices[product]
            p_base = COST_PRICE[product]

            # Elasticity response
            price_ratio = p / p_base
            elastic_factor = math.pow(price_ratio, ELASTICITY[product])

            # Log-normal noise (mean=1, std≈0.08) — prevents identical answers getting same result
            noise = self._rng.lognormvariate(0, 0.08)
            noise_log[product] = round(noise, 4)

            # Demand shock multiplier
            shock = self._demand_shock_active.get(product, 1.0)

            raw_demand = q_base * elastic_factor * noise * shock

            # Clamp demand: min 0, max 2x baseline
            demand[product] = max(0.0, min(raw_demand, q_base * 2.0))

        self._last_noise = noise_log
        return demand

    def _fulfill_sales(
        self, demand: Dict[str, float]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Sell from oldest batch first (FIFO). Return (sales, stockout)."""
        sales = {}
        stockout = {}
        for product in PRODUCTS:
            requested = int(round(demand[product]))
            available = sum(qty for qty, _ in self.inventory_batches[product])
            sold = min(requested, available)
            short = max(0, requested - available)
            sales[product] = sold
            stockout[product] = short

            # Deduct from FIFO batches
            remaining_to_sell = sold
            new_batches = []
            for qty, expiry_day in self.inventory_batches[product]:
                if remaining_to_sell <= 0:
                    new_batches.append((qty, expiry_day))
                elif qty <= remaining_to_sell:
                    remaining_to_sell -= qty
                    # batch fully sold
                else:
                    new_batches.append((qty - remaining_to_sell, expiry_day))
                    remaining_to_sell = 0
            self.inventory_batches[product] = new_batches

        return sales, stockout

    def _calc_profit(
        self, sales: Dict[str, int], prices: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        per_product = {}
        for product in PRODUCTS:
            revenue = sales[product] * prices[product]
            cost = sales[product] * COST_PRICE[product]
            p = revenue - cost
            per_product[product] = round(p, 2)
            total += p
        return round(total, 2), per_product

    def _do_restock(self) -> Dict[str, int]:
        restocked = {p: 0 for p in PRODUCTS}
        for product in PRODUCTS:
            if self.restock_next[product] <= self.day:
                qty = RESTOCK_QTY[product]
                expiry_day = self.day + SHELF_LIFE_DAYS[product]
                self.inventory_batches[product].append((qty, expiry_day))
                restocked[product] = qty
                self.restock_next[product] = self.day + RESTOCK_INTERVAL[product]
        return restocked

    def _update_inventory_totals(self):
        self.inventory = {
            p: sum(qty for qty, _ in self.inventory_batches[p])
            for p in PRODUCTS
        }

    def _calc_reward(
        self,
        profit: float,
        expired: Dict[str, int],
        stockout: Dict[str, int],
        sales: Dict[str, int],
        prices: Dict[str, float],
    ) -> DarkStoreReward:
        """
        Reward function:
          - Core: normalized profit signal
          - Spoilage penalty: proportional to wasted units (cost)
          - Stockout penalty: proportional to unmet demand (lost revenue opportunity)
          - Below-cost penalty: heavy penalty for pricing below cost
          - Normalized to [0, 1]
        """
        # ── Spoilage penalty ──
        spoilage_cost = sum(
            expired[p] * COST_PRICE[p] for p in PRODUCTS
        )
        spoilage_penalty = -spoilage_cost * 0.5  # penalize half the cost of expired goods

        # ── Stockout penalty ──
        stockout_penalty = -sum(
            stockout[p] * (prices[p] - COST_PRICE[p]) * 0.3
            for p in PRODUCTS
            if prices[p] > COST_PRICE[p]
        )

        # ── Below-cost penalty (hard disincentive) ──
        below_cost_penalty = 0.0
        for p in PRODUCTS:
            if prices[p] < COST_PRICE[p]:
                margin_loss = (COST_PRICE[p] - prices[p]) * sales[p]
                below_cost_penalty -= margin_loss * 2.0

        total_raw = profit + spoilage_penalty + stockout_penalty + below_cost_penalty

        # ── Normalize to [0, 1] ──
        # Max possible single-day profit (generous upper bound)
        max_possible = sum(
            BASELINE_DEMAND[p] * COST_PRICE[p] * 0.7
            for p in self.active_products
        )
        # Worst possible (all stock spoils, heavy penalty)
        min_possible = -sum(
            RESTOCK_QTY[p] * COST_PRICE[p] for p in self.active_products
        )

        if max_possible == min_possible:
            normalized = 0.5
        else:
            normalized = (total_raw - min_possible) / (max_possible - min_possible)

        normalized = min(max(normalized, 0.0), 1.0)

        components = {
            p: round(
                sales[p] * (prices[p] - COST_PRICE[p])
                - expired[p] * COST_PRICE[p] * 0.5
                - stockout[p] * max(0, prices[p] - COST_PRICE[p]) * 0.3,
                2,
            )
            for p in PRODUCTS
        }

        return DarkStoreReward(
            total_reward=round(normalized, 4),
            raw_profit=profit,
            spoilage_penalty=round(spoilage_penalty, 2),
            stockout_penalty=round(stockout_penalty, 2),
            components=components,
        )

    def _make_observation(self) -> DarkStoreObservation:
        # Expiry buckets: how many units expire in 1, 2, 3 days
        expiry_units: Dict[str, List[int]] = {}
        for product in PRODUCTS:
            buckets = [0, 0, 0]
            for qty, exp_day in self.inventory_batches[product]:
                days_left = exp_day - self.day
                if 1 <= days_left <= 3:
                    buckets[days_left - 1] += qty
            expiry_units[product] = buckets

        # Noisy demand forecast (±10%)
        demand_forecast = {}
        for p in PRODUCTS:
            noise = self._rng.uniform(0.9, 1.1)
            demand_forecast[p] = round(BASELINE_DEMAND[p] * noise, 1)

        # Restock schedule
        restock_schedule = {
            p: max(0, self.restock_next[p] - self.day)
            for p in PRODUCTS
        }

        return DarkStoreObservation(
            day=self.day,
            inventory=dict(self.inventory),
            expiry_units=expiry_units,
            cost_prices=dict(COST_PRICE),
            historical_sales=self.history_sales[-5:],
            historical_prices=self.history_prices[-5:],
            demand_forecast=demand_forecast,
            task_name=self.task_name,
            days_remaining=self.max_days - self.day,
            restock_schedule=restock_schedule,
        )


# ──────────────────────────── Task Graders ──────────────────────────────────

class EasyPricingGrader:
    """Grader for easy_pricing: was the agent profitable and avoided spoilage?"""

    def grade(self, env: DarkStoreEnv) -> float:
        score = env.compute_episode_score()
        # Easy task: give bonus if cumulative profit is positive
        if env.cumulative_profit > 0:
            bonus = min(0.1, env.cumulative_profit / 50000)
            score = min(1.0, score + bonus)
        return round(score, 4)


class MediumPricingGrader:
    """Grader for medium_pricing: multi-product efficiency."""

    def grade(self, env: DarkStoreEnv) -> float:
        base_score = env.compute_episode_score()

        # Penalty if any product never got sold (agent ignored it)
        product_sold = {p: False for p in PRODUCTS}
        for record in env.history_sales:
            for p in PRODUCTS:
                if record.get(p, 0) > 0:
                    product_sold[p] = True

        neglect_penalty = sum(0.1 for p in PRODUCTS if not product_sold[p])
        score = max(0.0, base_score - neglect_penalty)
        return round(score, 4)


class HardPricingGrader:
    """Grader for hard_pricing: adaptive pricing under shocks, strict spoilage."""

    def grade(self, env: DarkStoreEnv) -> float:
        base_score = env.compute_episode_score()

        # Bonus for positive cumulative profit at scale
        profit_bonus = 0.0
        if env.cumulative_profit > 0:
            profit_bonus = min(0.15, env.cumulative_profit / 500000)

        score = min(1.0, base_score + profit_bonus)
        return round(score, 4)


GRADERS = {
    "easy_pricing": EasyPricingGrader(),
    "medium_pricing": MediumPricingGrader(),
    "hard_pricing": HardPricingGrader(),
}

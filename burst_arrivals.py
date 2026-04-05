"""
Burst Arrival Experiment — Addresses W5
========================================
Tests whether the DQN outperforms FCFS when vessel arrivals are
clustered (burst pattern) rather than uniformly distributed.

Hypothesis: FCFS degrades under burst arrivals because it cannot
anticipate cascading delays. The DQN's sequential decision-making
should provide an advantage when immediate greedy dispatching
creates bottlenecks.

Generates 3 arrival patterns for 40 vessels, 16 berths:
  1. Uniform: arrivals spread evenly (current test — FCFS wins)
  2. Moderate burst: 3 clusters of 13-14 vessels
  3. Heavy burst: 2 clusters of 20 vessels

USAGE:
    python run_burst_arrivals.py

REQUIRES:
    hamburg_large_bap_berths.csv  (berth definitions)
    hamburg_large_bap_vessels_with_weather.csv  (for vessel properties)

    OR: will generate synthetic vessels if CSV not found
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import time
import os
from collections import deque
from scipy import stats

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

NUM_VESSELS = 40
NUM_BERTHS = 16


# ============================================================
# ENVIRONMENT
# ============================================================
class BurstBAPEnv:
    """
    BAP environment that accepts custom arrival times.
    Uses the same cost structure as your DQN code.
    """
    def __init__(self, arrivals, handling_times, num_berths):
        self.num_vessels = len(arrivals)
        self.num_berths = num_berths
        self.arrivals = np.array(arrivals, dtype=float)
        self.handling_times = np.array(handling_times, dtype=float)

        self.c_w = 100.0
        self.c_o = 500.0
        self.c_e = 50.0
        self.e_wait = 2.0
        self.e_handle = 5.0
        self.weather_factors = [0.0, 0.2, 0.5]

        self.feasibility = np.ones((self.num_vessels, self.num_berths), dtype=bool)
        self.action_space_size = self.num_vessels * self.num_berths
        self.state_dim = self.num_berths + self.num_vessels + 1

    def reset(self):
        self.assigned = set()
        self.berth_times = np.zeros(self.num_berths)
        self.weather = 0
        return self._state()

    def _state(self):
        flags = [1 if i in self.assigned else 0 for i in range(self.num_vessels)]
        return np.concatenate([self.berth_times / 100.0, flags, [self.weather]])

    def get_valid_actions(self):
        return [v * self.num_berths + b
                for v in range(self.num_vessels) if v not in self.assigned
                for b in range(self.num_berths) if self.feasibility[v, b]]

    def step(self, action):
        v = action // self.num_berths
        b = action % self.num_berths

        self.weather = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        delta = self.weather_factors[self.weather]

        a_v = self.arrivals[v]
        h_v = self.handling_times[v]
        h_tilde = h_v * (1.0 + delta)

        start = max(self.berth_times[b], a_v)
        wait = start - a_v
        self.berth_times[b] = start + h_tilde
        self.assigned.add(v)

        cost = (self.c_w + self.c_e * self.e_wait) * wait + \
               (self.c_o + self.c_e * self.e_handle) * h_tilde

        done = len(self.assigned) == self.num_vessels
        return self._state(), cost / 1000.0, done, cost


# ============================================================
# DQN
# ============================================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, cap=50000):
        self.buffer = deque(maxlen=cap)

    def push(self, s, a, c, ns, d):
        self.buffer.append((s, a, c, ns, d))

    def sample(self, bs):
        batch = random.sample(self.buffer, bs)
        s, a, c, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(c), np.array(ns), np.array(d)

    def __len__(self):
        return len(self.buffer)


# ============================================================
# ARRIVAL PATTERN GENERATORS
# ============================================================
def generate_uniform_arrivals(n, horizon=600):
    """Uniform: vessels spread evenly over horizon."""
    return sorted(np.random.uniform(0, horizon, n))


def generate_moderate_burst(n, n_clusters=3, cluster_spread=5, gap=80):
    """Moderate burst: n_clusters clusters with gap between them."""
    arrivals = []
    per_cluster = n // n_clusters
    remainder = n - per_cluster * n_clusters
    for c in range(n_clusters):
        center = c * (gap + cluster_spread * per_cluster)
        count = per_cluster + (1 if c < remainder else 0)
        for j in range(count):
            arrivals.append(center + np.random.uniform(0, cluster_spread))
    return sorted(arrivals)


def generate_heavy_burst(n, n_clusters=2, cluster_spread=3, gap=150):
    """Heavy burst: 2 tight clusters with large gap."""
    arrivals = []
    per_cluster = n // n_clusters
    remainder = n - per_cluster * n_clusters
    for c in range(n_clusters):
        center = c * (gap + cluster_spread * per_cluster)
        count = per_cluster + (1 if c < remainder else 0)
        for j in range(count):
            arrivals.append(center + np.random.uniform(0, cluster_spread))
    return sorted(arrivals)


def generate_handling_times(n, seed=42):
    """Generate realistic handling times (mix of vessel types)."""
    rng = np.random.RandomState(seed)
    times = []
    for i in range(n):
        vtype = rng.choice(['feeder', 'panamax', 'mother'], p=[0.4, 0.35, 0.25])
        if vtype == 'feeder':
            times.append(rng.uniform(4, 8))
        elif vtype == 'panamax':
            times.append(rng.uniform(10, 18))
        else:
            times.append(rng.uniform(20, 35))
    return times


# ============================================================
# EVALUATION METHODS
# ============================================================
def evaluate_fcfs(env, n_evals=100):
    """FCFS: assign in arrival order to earliest available berth."""
    costs = []
    for _ in range(n_evals):
        env.reset()
        order = np.argsort(env.arrivals)
        total = 0
        for v in order:
            best_b = np.argmin(env.berth_times)
            action = v * env.num_berths + best_b
            _, _, _, c = env.step(action)
            total += c
        costs.append(total)
    return np.mean(costs), np.std(costs)


def evaluate_dqn(env, policy_net, n_evals=100):
    """Trained DQN policy."""
    costs = []
    for _ in range(n_evals):
        state = env.reset()
        total = 0
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            with torch.no_grad():
                q = policy_net(torch.FloatTensor(state))
                mask = torch.full((env.action_space_size,), float('inf'))
                for a in valid:
                    mask[a] = 0
                action = (q + mask).argmin().item()
            state, _, done, c = env.step(action)
            total += c
        costs.append(total)
    return np.mean(costs), np.std(costs)


def train_dqn_on_pattern(arrivals, handling_times, num_berths,
                          episodes=3000, label=""):
    """Train a DQN on a specific arrival pattern."""
    env = BurstBAPEnv(arrivals, handling_times, num_berths)
    policy_net = DQN(env.state_dim, env.action_space_size)
    target_net = DQN(env.state_dim, env.action_space_size)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
    memory = ReplayBuffer(50000)
    epsilon = 1.0

    print(f"  Training DQN ({label}): {episodes} episodes...", end=" ", flush=True)
    t0 = time.time()

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            if random.random() < epsilon:
                action = random.choice(valid)
            else:
                with torch.no_grad():
                    q = policy_net(torch.FloatTensor(state))
                    mask = torch.full((env.action_space_size,), float('inf'))
                    for a in valid:
                        mask[a] = 0
                    action = (q + mask).argmin().item()

            next_state, cost_s, done, _ = env.step(action)
            memory.push(state, action, cost_s, next_state, done)
            state = next_state

            if len(memory) >= 64:
                s, a, c, ns, d = memory.sample(64)
                s_t = torch.FloatTensor(s)
                a_t = torch.LongTensor(a).unsqueeze(1)
                c_t = torch.FloatTensor(c).unsqueeze(1)
                ns_t = torch.FloatTensor(ns)
                d_t = torch.FloatTensor(d).unsqueeze(1)
                q_cur = policy_net(s_t).gather(1, a_t)
                with torch.no_grad():
                    q_nxt = target_net(ns_t).min(1)[0].unsqueeze(1)
                    q_tgt = c_t + 0.99 * q_nxt * (1 - d_t)
                loss = nn.MSELoss()(q_cur, q_tgt)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        epsilon = max(0.05, epsilon * 0.999)
        if (ep + 1) % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    elapsed = time.time() - t0
    print(f"done ({elapsed:.0f}s)")
    return policy_net


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("W5: Burst Arrival Experiment")
    print(f"    {NUM_VESSELS} vessels, {NUM_BERTHS} berths")
    print("=" * 70)

    handling = generate_handling_times(NUM_VESSELS, seed=SEED)

    patterns = {
        'Uniform': generate_uniform_arrivals(NUM_VESSELS, horizon=600),
        'Moderate Burst': generate_moderate_burst(NUM_VESSELS, n_clusters=3),
        'Heavy Burst': generate_heavy_burst(NUM_VESSELS, n_clusters=2),
    }

    # Show arrival distributions
    print("\nArrival Patterns:")
    for name, arr in patterns.items():
        print(f"  {name}: range [{min(arr):.1f}, {max(arr):.1f}], "
              f"mean spacing = {np.mean(np.diff(arr)):.1f}h")

    results = {}

    for name, arrivals in patterns.items():
        print(f"\n{'='*60}")
        print(f"Pattern: {name}")
        print(f"{'='*60}")

        env = BurstBAPEnv(arrivals, handling, NUM_BERTHS)

        # FCFS
        fcfs_mean, fcfs_std = evaluate_fcfs(env)
        print(f"  FCFS: {fcfs_mean:,.0f} ± {fcfs_std:,.0f}")

        # Train DQN for this pattern
        policy = train_dqn_on_pattern(
            arrivals, handling, NUM_BERTHS,
            episodes=3000, label=name
        )

        # Evaluate DQN
        dqn_mean, dqn_std = evaluate_dqn(env, policy)
        gap = (dqn_mean - fcfs_mean) / fcfs_mean * 100
        print(f"  DQN:  {dqn_mean:,.0f} ± {dqn_std:,.0f}  (Gap vs FCFS: {gap:+.1f}%)")

        results[name] = {
            'fcfs_mean': fcfs_mean, 'fcfs_std': fcfs_std,
            'dqn_mean': dqn_mean, 'dqn_std': dqn_std,
            'gap': gap,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Pattern':<20} {'FCFS':>15} {'DQN':>15} {'Gap':>10}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<20} {r['fcfs_mean']:>12,.0f}   {r['dqn_mean']:>12,.0f}   "
              f"{r['gap']:>+8.1f}%")

    # Statistical test
    print("\n" + "=" * 70)
    print("KEY QUESTION: Does the gap shrink (or reverse) under burst arrivals?")
    print("=" * 70)

    uniform_gap = results['Uniform']['gap']
    heavy_gap = results['Heavy Burst']['gap']

    if heavy_gap < uniform_gap:
        print(f"  YES: DQN gap reduces from {uniform_gap:+.1f}% (uniform) "
              f"to {heavy_gap:+.1f}% (heavy burst)")
        if heavy_gap < 0:
            print(f"  → DQN OUTPERFORMS FCFS under heavy burst arrivals!")
        else:
            print(f"  → Gap reduced by {uniform_gap - heavy_gap:.1f} percentage points")
    else:
        print(f"  NO: DQN gap does not improve under burst arrivals")
        print(f"  Uniform: {uniform_gap:+.1f}%, Heavy Burst: {heavy_gap:+.1f}%")

    print("\n" + "=" * 70)
    print("LaTeX TABLE ROW (for paper):")
    print("=" * 70)
    for name, r in results.items():
        print(f"{name} & ${r['fcfs_mean']:,.0f} \\pm {r['fcfs_std']:,.0f}$ "
              f"& ${r['dqn_mean']:,.0f} \\pm {r['dqn_std']:,.0f}$ "
              f"& ${r['gap']:+.1f}\\%$ \\\\")

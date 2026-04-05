import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import time
import csv
from collections import deque

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


# ============================================================
# ENVIRONMENT
# ============================================================
class RealDataBAPEnv:
    def __init__(self, vessels_csv, berths_csv):
        self.vessels_df = pd.read_csv(vessels_csv)
        self.berths_df = pd.read_csv(berths_csv)
        self.num_vessels = len(self.vessels_df)
        self.num_berths = len(self.berths_df)

        self.vessels_df['arrival_time'] = pd.to_datetime(self.vessels_df['arrival_time'])
        min_time = self.vessels_df['arrival_time'].min()
        self.vessels_df['arrival_hours'] = (
            self.vessels_df['arrival_time'] - min_time
        ).dt.total_seconds() / 3600.0

        self.has_physical_dims = (
            'L_v' in self.vessels_df.columns and 'D_v' in self.vessels_df.columns
        )
        self.handling_col = 'H_v0' if 'H_v0' in self.vessels_df.columns else 'handling_time'

        self.feasibility = np.ones((self.num_vessels, self.num_berths), dtype=bool)
        if self.has_physical_dims:
            for v in range(self.num_vessels):
                for b in range(self.num_berths):
                    v_len = self.vessels_df.iloc[v]['L_v']
                    v_draft = self.vessels_df.iloc[v]['D_v']
                    b_len = self.berths_df.iloc[b]['berth_length']
                    b_depth = self.berths_df.iloc[b]['berth_depth']
                    if v_len > b_len or v_draft > b_depth:
                        self.feasibility[v, b] = False

        self.c_w = 100.0
        self.c_o = 500.0
        self.c_e = 50.0
        self.weather_factors = [0.0, 0.2, 0.5]
        self.action_space_size = self.num_vessels * self.num_berths
        self.state_dim = self.num_berths + self.num_vessels + 1

    def reset(self):
        self.assigned_vessels = set()
        self.berth_times = np.zeros(self.num_berths)
        self.weather = 0
        return self._get_state()

    def _get_state(self):
        vessel_status = [1 if i in self.assigned_vessels else 0
                         for i in range(self.num_vessels)]
        return np.concatenate([self.berth_times, vessel_status, [self.weather]])

    def get_valid_actions(self):
        valid = []
        for v in range(self.num_vessels):
            if v not in self.assigned_vessels:
                for b in range(self.num_berths):
                    if self.feasibility[v, b]:
                        valid.append(v * self.num_berths + b)
        return valid

    def step(self, action_idx):
        vessel = action_idx // self.num_berths
        berth = action_idx % self.num_berths
        self.weather = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        delta_w = self.weather_factors[self.weather]

        a_v = self.vessels_df.iloc[vessel]['arrival_hours']
        h_v = self.vessels_df.iloc[vessel][self.handling_col]
        e_wait, e_handle = 2.0, 5.0
        h_tilde = h_v * (1.0 + delta_w)
        start_time = max(self.berth_times[berth], a_v)
        W_v = start_time - a_v
        self.berth_times[berth] = start_time + h_tilde
        self.assigned_vessels.add(vessel)

        cost = (self.c_w + self.c_e * e_wait) * W_v + \
               (self.c_o + self.c_e * e_handle) * h_tilde
        done = len(self.assigned_vessels) == self.num_vessels
        return self._get_state(), cost / 1000.0, done, cost


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
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, c, ns, d):
        self.buffer.append((s, a, c, ns, d))

    def sample(self, bs):
        batch = random.sample(self.buffer, bs)
        s, a, c, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(c), np.array(ns), np.array(d)

    def __len__(self):
        return len(self.buffer)


# ============================================================
# FCFS BASELINE
# ============================================================
def evaluate_fcfs(env, n_evals=100):
    """FCFS with stochastic weather, averaged over n_evals runs."""
    costs = []
    for _ in range(n_evals):
        env.reset()
        # Sort by arrival order
        order = np.argsort(env.vessels_df['arrival_hours'].values)
        total_cost = 0
        for v in order:
            # Assign to earliest available feasible berth
            best_b, best_t = -1, float('inf')
            for b in range(env.num_berths):
                if env.feasibility[v, b] and env.berth_times[b] < best_t:
                    best_b, best_t = b, env.berth_times[b]
            if best_b >= 0:
                action = v * env.num_berths + best_b
                _, _, _, c = env.step(action)
                total_cost += c
        costs.append(total_cost)
    return np.mean(costs), np.std(costs)


# ============================================================
# TRAINING
# ============================================================
def train_extended(vessels_csv, berths_csv, num_episodes=10000):
    env = RealDataBAPEnv(vessels_csv, berths_csv)
    print(f"Environment: {env.num_vessels} vessels, {env.num_berths} berths")
    print(f"State dim: {env.state_dim}, Action dim: {env.action_space_size}")

    policy_net = DQN(env.state_dim, env.action_space_size)
    target_net = DQN(env.state_dim, env.action_space_size)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
    memory = ReplayBuffer(50000)
    epsilon = 1.0
    gamma = 0.99
    batch_size = 64

    episode_costs = []
    checkpoints = {}

    print(f"\nTraining {num_episodes} episodes...")
    t0 = time.time()

    for ep in range(num_episodes):
        state = env.reset()
        total_cost = 0
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
                    # Mask invalid actions
                    mask = torch.full((env.action_space_size,), float('inf'))
                    for a in valid:
                        mask[a] = 0
                    q = q + mask
                    action = q.argmin().item()

            next_state, cost_scaled, done, cost_raw = env.step(action)
            total_cost += cost_raw
            memory.push(state, action, cost_scaled, next_state, done)
            state = next_state

            if len(memory) >= batch_size:
                s, a, c, ns, d = memory.sample(batch_size)
                s_t = torch.FloatTensor(s)
                a_t = torch.LongTensor(a).unsqueeze(1)
                c_t = torch.FloatTensor(c).unsqueeze(1)
                ns_t = torch.FloatTensor(ns)
                d_t = torch.FloatTensor(d).unsqueeze(1)

                q_cur = policy_net(s_t).gather(1, a_t)
                with torch.no_grad():
                    q_next = target_net(ns_t).min(1)[0].unsqueeze(1)
                    q_target = c_t + gamma * q_next * (1 - d_t)

                loss = nn.MSELoss()(q_cur, q_target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        epsilon = max(0.05, epsilon * 0.9995)
        if (ep + 1) % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_costs.append(total_cost)

        # Save checkpoints at key points
        if (ep + 1) in [2000, 5000, 10000]:
            checkpoints[ep + 1] = {
                'mean_last_500': np.mean(episode_costs[-500:]),
                'std_last_500': np.std(episode_costs[-500:]),
                'epsilon': epsilon,
            }

        if (ep + 1) % 1000 == 0:
            avg = np.mean(episode_costs[-500:])
            elapsed = time.time() - t0
            print(f"  Ep {ep+1}: avg cost (500-ep window) = {avg:,.0f}, "
                  f"ε = {epsilon:.4f}, time = {elapsed:.0f}s")

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f} min)")

    return policy_net, episode_costs, checkpoints


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("W2: Extended Training (10,000 episodes) on Large Instance")
    print("=" * 65)

    vessels_csv = 'hamburg_large_bap_vessels_with_weather.csv'
    berths_csv = 'hamburg_large_bap_berths.csv'

    policy_net, costs, checkpoints = train_extended(
        vessels_csv, berths_csv, num_episodes=10000
    )

    # Save learning curve
    with open('learning_curve_10k.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'cost'])
        for i, c in enumerate(costs):
            w.writerow([i + 1, c])
    print("Saved: learning_curve_10k.csv")

    # FCFS baseline
    env = RealDataBAPEnv(vessels_csv, berths_csv)
    fcfs_mean, fcfs_std = evaluate_fcfs(env)
    print(f"\nFCFS baseline: {fcfs_mean:,.0f} ± {fcfs_std:,.0f}")

    # Plateau analysis
    print("\n" + "=" * 65)
    print("PLATEAU ANALYSIS")
    print("=" * 65)
    print(f"\n{'Checkpoint':<12} {'Avg Cost (500ep)':>18} {'Std':>12} {'Gap vs FCFS':>14}")
    print("-" * 60)
    for ep, data in sorted(checkpoints.items()):
        gap = (data['mean_last_500'] - fcfs_mean) / fcfs_mean * 100
        print(f"Ep {ep:<8} {data['mean_last_500']:>18,.0f} {data['std_last_500']:>12,.0f} "
              f"{gap:>+13.1f}%")

    # Check if still improving
    if len(checkpoints) >= 2:
        eps = sorted(checkpoints.keys())
        c1 = checkpoints[eps[-2]]['mean_last_500']
        c2 = checkpoints[eps[-1]]['mean_last_500']
        improvement = (c1 - c2) / c1 * 100
        print(f"\nImprovement from ep {eps[-2]} to {eps[-1]}: {improvement:+.2f}%")
        if abs(improvement) < 1.0:
            print("→ PLATEAUED: <1% improvement confirms architectural ceiling")
        else:
            print("→ STILL IMPROVING: consider training longer")

    # TikZ coordinates for latex
    print("\n" + "=" * 65)
    print("TikZ COORDINATES (rolling 200-episode average):")
    print("=" * 65)
    window = 200
    roll = [np.mean(costs[max(0, i-window+1):i+1]) for i in range(len(costs))]
    sample_points = list(range(199, 10000, 200))
    coords = [f"({i+1},{roll[i]:.0f})" for i in sample_points]
    print("\\addplot[thick, blue, smooth] coordinates {")
    for j in range(0, len(coords), 5):
        print("    " + " ".join(coords[j:j+5]))
    print("};")
    print(f"% FCFS reference line: {fcfs_mean:.0f}")

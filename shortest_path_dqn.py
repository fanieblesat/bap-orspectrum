import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque

# ==========================================
# 1. The Neural Network
# ==========================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # 256 neurons to handle the larger state spaces of Medium/Large datasets
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================
# 2. The Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, cost, next_state, done):
        self.buffer.append((state, action, cost, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, costs, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(costs),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 3. The Universal Real-Data Environment
# ==========================================
class RealDataStochasticBAPEnv:
    def __init__(self, vessels_csv, berths_csv):
        # Load Datasets
        self.vessels_df = pd.read_csv(vessels_csv)
        self.berths_df = pd.read_csv(berths_csv)

        self.num_vessels = len(self.vessels_df)
        self.num_berths = len(self.berths_df)

        # Calculate Relative Arrival Times
        self.vessels_df['arrival_time'] = pd.to_datetime(self.vessels_df['arrival_time'])
        min_time = self.vessels_df['arrival_time'].min()
        self.vessels_df['arrival_hours'] = (self.vessels_df['arrival_time'] - min_time).dt.total_seconds() / 3600.0

        # Determine column naming conventions from the CSV
        self.has_physical_dims = 'L_v' in self.vessels_df.columns and 'D_v' in self.vessels_df.columns
        self.handling_col = 'H_v0' if 'H_v0' in self.vessels_df.columns else 'handling_time'

        # Pre-calculate physical feasibility matrix
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

        # Cost Parameters
        self.c_w = 100.0
        self.c_o = 500.0
        self.c_e = 50.0
        self.weather_factors = [0.0, 0.2, 0.5] # Clear, Moderate, Severe

        self.action_space_size = self.num_vessels * self.num_berths
        self.state_dim = self.num_berths + self.num_vessels + 1

    def reset(self):
        self.assigned_vessels = set()
        self.berth_times = np.zeros(self.num_berths)
        self.weather = 0
        return self._get_state()

    def _get_state(self):
        vessel_status = [1 if i in self.assigned_vessels else 0 for i in range(self.num_vessels)]
        return np.concatenate([self.berth_times, vessel_status, [self.weather]])

    def get_valid_actions(self):
        valid_actions = []
        for v in range(self.num_vessels):
            if v not in self.assigned_vessels:
                for b in range(self.num_berths):
                    if self.feasibility[v, b]:
                        valid_actions.append(v * self.num_berths + b)
        return valid_actions

    def step(self, action_idx):
        vessel = action_idx // self.num_berths
        berth = action_idx % self.num_berths

        self.weather = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        delta_w = self.weather_factors[self.weather]

        # Pull actual Real-World Data using detected columns
        a_v = self.vessels_df.iloc[vessel]['arrival_hours']
        h_v = self.vessels_df.iloc[vessel][self.handling_col]

        e_wait = 2.0
        e_handle = 5.0

        h_tilde = h_v * (1.0 + delta_w)
        start_time = max(self.berth_times[berth], a_v)
        W_v = start_time - a_v

        self.berth_times[berth] = start_time + h_tilde
        self.assigned_vessels.add(vessel)

        cost_wait = (self.c_w + self.c_e * e_wait) * W_v
        cost_handle = (self.c_o + self.c_e * e_handle) * h_tilde
        total_cost = cost_wait + cost_handle

        done = len(self.assigned_vessels) == self.num_vessels

        scaled_cost = total_cost / 1000.0

        return self._get_state(), scaled_cost, done

# ==========================================
# 4. The Agent & Training Loop
# ==========================================
def train_dqn():
    # Currently set to the MEDIUM dataset
    env = RealDataStochasticBAPEnv(
        vessels_csv='hamburg_large_bap_vessels_with_weather.csv',
        berths_csv='hamburg_large_bap_berths.csv'
    )

    print(f"Loaded Environment: {env.num_vessels} Vessels, {env.num_berths} Berths")

    policy_net = DQN(env.state_dim, env.action_space_size)
    target_net = DQN(env.state_dim, env.action_space_size)
    target_net.load_state_dict(policy_net.state_dict())

    # Optimizer LR adjusted for stability on larger state spaces
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
    memory = ReplayBuffer(capacity=30000)

    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    batch_size = 64

    # Increased episodes for the medium/large dataset complexity
    for episode in range(2000):
        state = env.reset()
        total_real_cost = 0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()

            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_tensor)[0]
                    for i in range(env.action_space_size):
                        if i not in valid_actions:
                            q_values[i] = float('inf')
                    action = q_values.argmin().item()

            next_state, scaled_cost, done = env.step(action)
            memory.push(state, action, scaled_cost, next_state, done)

            state = next_state
            total_real_cost += (scaled_cost * 1000.0)

            if len(memory) > batch_size:
                s, a, c, s_prime, d = memory.sample(batch_size)

                s = torch.FloatTensor(s)
                a = torch.LongTensor(a).unsqueeze(1)
                c = torch.FloatTensor(c).unsqueeze(1)
                s_prime = torch.FloatTensor(s_prime)
                d = torch.FloatTensor(d).unsqueeze(1)

                q_values = policy_net(s).gather(1, a)

                with torch.no_grad():
                    next_q_values = target_net(s_prime).min(1)[0].unsqueeze(1)
                    target_q = c + gamma * next_q_values * (1 - d)

                loss = nn.MSELoss()(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Real Cost: {total_real_cost:.2f}, Epsilon: {epsilon:.2f}")
            # ==========================================
    # --- ADD THIS RIGHT HERE ---
    # After the loop finishes, run the strict evaluation!
    # ==========================================
    print("\nTraining Complete! Running strictly greedy evaluation on stochastic weather...")

    # Call the evaluation function using the fully trained policy_net
    final_eval_cost = evaluate_dqn(env, policy_net, num_episodes=100)

    return final_eval_cost

# Make sure the evaluate_dqn function is pasted somewhere in your file!
def evaluate_dqn(env, policy_net, num_episodes=100):
    print("\n--- STARTING DETERMINISTIC EVALUATION ---")
    total_eval_cost = 0.0

    for episode in range(num_episodes):
        state = env.reset()
        episode_cost = 0.0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()

            # STRICT GREEDY POLICY (NO EPSILON RANDOMNESS)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)[0]
                for i in range(env.action_space_size):
                    if i not in valid_actions:
                        q_values[i] = float('inf')
                action = q_values.argmin().item()

            next_state, scaled_cost, done = env.step(action)
            state = next_state
            episode_cost += (scaled_cost * 1000.0)

        total_eval_cost += episode_cost

    avg_cost = total_eval_cost / num_episodes
    print(f"TRUE EVALUATION AVERAGE COST: {avg_cost:,.2f}")
    return avg_cost

if __name__ == "__main__":
    train_dqn()
    def evaluate_dqn(env, policy_net, num_episodes=100):
      print("\n--- STARTING DETERMINISTIC EVALUATION ---")
      total_eval_cost = 0.0

      for episode in range(num_episodes):
          state = env.reset()
          episode_cost = 0.0
          done = False

          while not done:
              valid_actions = env.get_valid_actions()

              # STRICT GREEDY POLICY (NO EPSILON RANDOMNESS)
              with torch.no_grad():
                  state_tensor = torch.FloatTensor(state).unsqueeze(0)
                  q_values = policy_net(state_tensor)[0]
                  for i in range(env.action_space_size):
                      if i not in valid_actions:
                          q_values[i] = float('inf')
                  action = q_values.argmin().item()

              next_state, scaled_cost, done = env.step(action)
              state = next_state
              episode_cost += (scaled_cost * 1000.0)

          total_eval_cost += episode_cost

      avg_cost = total_eval_cost / num_episodes
      print(f"TRUE EVALUATION AVERAGE COST: {avg_cost:,.2f}")
      return avg_cost
import pandas as pd
import numpy as np

# --- LOAD DATA ---
df = pd.read_csv('hamburg_large_bap_vessels_with_weather.csv')
df['arrival_time'] = pd.to_datetime(df['arrival_time'])
df['arrival_hours'] = (df['arrival_time'] - df['arrival_time'].min()).dt.total_seconds() / 3600.0

arrival_times = df['arrival_hours'].tolist()
#handling_times = df['H_v(s)'].tolist()
num_vessels = len(df)
berths_df = pd.read_csv('hamburg_large_bap_berths.csv')  # match the instance
num_berths = len(berths_df)
handling_times = df['handling_time'].tolist()


# --- EXACT DQN WEIGHTS ---
cost_per_wait_hr = 200.0
cost_per_handle_hr = 750.0

# --- FCFS LOGIC ---
vessels = list(range(num_vessels))
vessels.sort(key=lambda v: arrival_times[v])

berth_available_times = [0.0] * num_berths
total_cost = 0.0

for v in vessels:
    best_berth = np.argmin(berth_available_times)
    start_time = max(arrival_times[v], berth_available_times[best_berth])

    waiting_time = start_time - arrival_times[v]

    # EXACT MATCH TO DQN COST
    cost = (waiting_time * cost_per_wait_hr) + (handling_times[v] * cost_per_handle_hr)

    total_cost += cost
    berth_available_times[best_berth] = start_time + handling_times[v]

print(f"\nTRUE FCFS COST: {total_cost:,.2f}")
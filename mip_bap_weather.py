import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# --- LOAD DATA ---
df = pd.read_csv('hamburg_bap_vessels_with_weather_small.csv')
df['arrival_time'] = pd.to_datetime(df['arrival_time'])
df['arrival_hours'] = (df['arrival_time'] - df['arrival_time'].min()).dt.total_seconds() / 3600.0

arrival_times = df['arrival_hours'].tolist()
handling_times = df['H_v0'].tolist()

num_vessels = len(df)
num_berths = 3
V = range(num_vessels)
B = range(num_berths)

# --- YOUR EXACT DQN WEIGHTS ---
cost_per_wait_hr = 200.0   # (c_w + c_e * e_wait)
cost_per_handle_hr = 750.0 # (c_o + c_e * e_handle)

# --- GUROBI MODEL ---
m = gp.Model("Berth_Allocation_MIP")
M = 10000

x = m.addVars(num_vessels, num_berths, vtype=GRB.BINARY, name="x")
y = m.addVars(num_vessels, num_vessels, vtype=GRB.BINARY, name="y")
t = m.addVars(num_vessels, vtype=GRB.CONTINUOUS, lb=0.0, name="t")

# EXACT MATCH TO DQN OBJECTIVE
m.setObjective(
    gp.quicksum(
        (t[v] - arrival_times[v]) * cost_per_wait_hr +
        (handling_times[v]) * cost_per_handle_hr
        for v in V
    ), GRB.MINIMIZE
)

# CONSTRAINTS
for v in V:
    m.addConstr(gp.quicksum(x[v, b] for b in B) == 1)
    m.addConstr(t[v] >= arrival_times[v])

for v in V:
    for w in V:
        if v != w:
            for b in B:
                m.addConstr(t[w] >= t[v] + handling_times[v] - M * (1 - y[v, w]) - M * (2 - x[v, b] - x[w, b]))
                m.addConstr(t[v] >= t[w] + handling_times[w] - M * y[v, w] - M * (2 - x[v, b] - x[w, b]))

m.setParam('TimeLimit', 3600)
m.optimize()

if m.status == GRB.OPTIMAL:
    print(f"\nTRUE MIP OPTIMAL COST: {m.objVal:,.2f}")
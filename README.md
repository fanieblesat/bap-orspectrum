# From Shortest Paths to Deep RL — Codebase

**Paper:** From Shortest Paths to Deep Reinforcement Learning: An Extended Graph-Based Formulation of the Berth Allocation Problem under Uncertainty  
**Venue:** Transportation Research Part B, Special Issue: Mathematical Foundations for Trustworthy AI Applications in Transportation Systems 
**Authors:** Fabricio Niebles-Atencio, Stefan Voß  
**Affiliation:** University of Hamburg

---

## Repository Structure

```
├── shortest_path_dqn.py      Main DQN agent: environment, network, training loop
├── mip_bap_weather.py         MIP exact solver (Gurobi) — Small instance
├── fcfs_baseline.py           FCFS baseline evaluation
├── ga_medium_large.py         Genetic Algorithm on Medium/Large instances
├── dqn_convergence.py         Training convergence plotting
├── extended_training.py       Extended 10K-episode training experiment
├── burst_arrivals.py          Burst arrival pattern experiment
│
├── data/                      Port of Hamburg instance data
│   ├── hamburg_bap_vessels_with_weather_small.csv    Small: 10 vessels
│   ├── hamburg_small_bap_berths.csv                  Small: 3 berths
│   ├── hamburg_medium_bap_vessels_with_weather.csv   Medium: 20 vessels
│   ├── hamburg_medium_bap_berths.csv                 Medium: 8 berths
│   ├── hamburg_large_bap_vessels_with_weather.csv    Large: 40 vessels
│   ├── hamburg_large_bap_berths.csv                  Large: 16 berths
│   └── hamburg_congested_*.csv                       Congestion analysis
│
└── notebooks/                 Original Jupyter notebooks (Google Colab)
    ├── Shortest_Path_DQN.ipynb
    ├── MIP_BAP_Weather.ipynb
    ├── FCFS.ipynb
    ├── DQN_Convergence.ipynb
    ├── GA_Medium_Large.ipynb
    ├── Extended_Training.ipynb
    └── Burst_Arrivals.ipynb
```

## Scripts → Paper Mapping

| Script | Paper Section | What it produces |
|--------|--------------|-----------------|
| `shortest_path_dqn.py` | §8.3–8.4 (Tables 2–3) | DQN training + evaluation on all instances |
| `mip_bap_weather.py` | §8.3 (Table 2) | MIP-proven optimal on Small (69,133) |
| `fcfs_baseline.py` | §8.3–8.4 (Tables 2–3) | FCFS baseline costs |
| `ga_medium_large.py` | §8.4 (Table 3) | GA on Medium (205,929) and Large (457,293) |
| `dqn_convergence.py` | §8.6 (Figure 5) | Training convergence visualization |
| `extended_training.py` | §8.6 (Limitations) | 10K-episode training — divergence evidence |
| `burst_arrivals.py` | §9 (Discussion) | Burst arrival pattern analysis |

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy pandas matplotlib scipy deap
# For MIP solver (optional):
pip install gurobipy  # Requires Gurobi license
```

### 2. Train the DQN (Large instance)

```bash
python shortest_path_dqn.py
```

This trains for 2,000 episodes on the Large instance (40 vessels, 16 berths) and outputs the evaluation cost.

### 3. Run baselines

```bash
python fcfs_baseline.py            # FCFS on Large instance
python mip_bap_weather.py          # MIP on Small instance (requires Gurobi)
python ga_medium_large.py          # GA on Medium and Large
```

### 4. Reproduce paper results

To reproduce all results in the paper, run the scripts in this order:

1. `mip_bap_weather.py` → Table 2, MIP column
2. `fcfs_baseline.py` → Tables 2–3, FCFS column
3. `shortest_path_dqn.py` → Tables 2–3, DQN column (modify CSV paths for each instance)
4. `ga_medium_large.py` → Table 3, GA column
5. `dqn_convergence.py` → Figure 5

### 5. Supplementary experiments (reviewer responses)

```bash
python extended_training.py        # W2: Shows divergence after ~5,500 episodes
python burst_arrivals.py           # W5: Tests DQN under clustered vessel arrivals
```

## Key Results

| Method | Small (10v, 3b) | Medium (20v, 8b) | Large (40v, 16b) |
|--------|----------------|------------------|------------------|
| MIP | 69,133 (optimal) | Timeout | Timeout |
| GA | 69,133 (0.0%) | 205,929 (-0.3%) | 457,293 (+0.6%) |
| FCFS | 69,977 (+1.2%) | 206,557 | 454,615 |
| DQN | 78,773 (+13.9%) | 224,754 (+8.8%) | 523,019 (+15.1%) |
| Tabular Q | 70,116 (+1.4%) | — | — |

## Data

All instances use real vessel data from the Port of Hamburg. Weather transitions follow a 3-state Markov chain (Clear/Moderate/Severe) calibrated to DWD Climate Data Center records (Station 01975).

## Citation

```bibtex
@article{niebles2025lsps,
  title={From Shortest Paths to Deep Reinforcement Learning: An Extended 
         Graph-Based Formulation of the Berth Allocation Problem under Uncertainty},
  author={Niebles-Atencio, Fabricio and Vo{\ss}, Stefan},
  journal={Transportation Research Part B},
  year={2026},
  note={Under revision}
}
```

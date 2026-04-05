import numpy as np
import pandas as pd
import random
import time


try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("ERROR: deap library not found. Please ensure it's installed.")
    exit(1)


def run_ga_on_instance(vessels_csv, berths_csv, pop_size=200, n_gen=200,
                       label="Instance"):
    """Run GA and return deterministic + stochastic costs."""
    df = pd.read_csv(vessels_csv)
    df['arrival_time'] = pd.to_datetime(df['arrival_time'])
    df['arrival_hours'] = (
        df['arrival_time'] - df['arrival_time'].min()
    ).dt.total_seconds() / 3600.0
    handling_col = 'H_v0' if 'H_v0' in df.columns else 'handling_time'

    berths_df = pd.read_csv(berths_csv)
    num_vessels = len(df)
    num_berths = len(berths_df)

    cost_wait = 200.0    # c_w + c_e * e_wait
    cost_handle = 750.0  # c_o + c_e * e_handle

    print(f"\n{'='*60}")
    print(f"GA on {label}: {num_vessels} vessels, {num_berths} berths")
    print(f"{'='*60}")

    def evaluate_schedule(individual):
        order = individual[:num_vessels]
        berth_assign = individual[num_vessels:]
        berth_available = [0.0] * num_berths
        total_cost = 0.0
        for pos in range(num_vessels):
            v = order[pos]
            b = berth_assign[pos] % num_berths
            a_v = df.iloc[v]['arrival_hours']
            h_v = df.iloc[v][handling_col]
            start = max(berth_available[b], a_v)
            wait = start - a_v
            cost = wait * cost_wait + h_v * cost_handle
            total_cost += cost
            berth_available[b] = start + h_v
        return (total_cost,)

    # DEAP setup
    if hasattr(creator, "FitnessMin_GA"):
        del creator.FitnessMin_GA
    if hasattr(creator, "Individual_GA"):
        del creator.Individual_GA

    creator.create("FitnessMin_GA", base.Fitness, weights=(-1.0,))
    creator.create("Individual_GA", list, fitness=creator.FitnessMin_GA)

    toolbox = base.Toolbox()

    def create_individual():
        order = list(np.random.permutation(num_vessels))
        berths = [random.randint(0, num_berths - 1) for _ in range(num_vessels)]
        return creator.Individual_GA(order + berths)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_schedule)

    def cx_order_and_berth(ind1, ind2):
        o1, o2 = ind1[:num_vessels], ind2[:num_vessels]
        tools.cxOrdered(o1, o2)
        b1, b2 = ind1[num_vessels:], ind2[num_vessels:]
        tools.cxUniform(b1, b2, indpb=0.5)
        ind1[:] = o1 + b1
        ind2[:] = o2 + b2
        return ind1, ind2

    def mut_swap_and_berth(individual):
        order = individual[:num_vessels]
        i, j = random.sample(range(num_vessels), 2)
        order[i], order[j] = order[j], order[i]
        if random.random() < 0.3:
            k = random.randint(0, num_vessels - 1)
            individual[num_vessels + k] = random.randint(0, num_berths - 1)
        individual[:num_vessels] = order
        return (individual,)

    toolbox.register("mate", cx_order_and_berth)
    toolbox.register("mutate", mut_swap_and_berth)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    print(f"Running GA: pop={pop_size}, generations={n_gen}")
    t0 = time.time()
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_gen,
        stats=stats, halloffame=hof, verbose=False
    )
    elapsed = time.time() - t0

    best_det = hof[0].fitness.values[0]
    print(f"  Deterministic cost: {best_det:,.2f}")
    print(f"  Time: {elapsed:.1f}s")

    # Stochastic evaluation (100 runs)
    best_order = hof[0][:num_vessels]
    best_berths = hof[0][num_vessels:]
    stoch_costs = []
    for _ in range(100):
        berth_avail = [0.0] * num_berths
        total_cost = 0.0
        for pos in range(num_vessels):
            v = best_order[pos]
            b = best_berths[pos] % num_berths
            a_v = df.iloc[v]['arrival_hours']
            h_v = df.iloc[v][handling_col]
            weather = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            h_tilde = h_v * (1.0 + [0.0, 0.2, 0.5][weather])
            start = max(berth_avail[b], a_v)
            wait = start - a_v
            cost = wait * cost_wait + h_tilde * cost_handle
            total_cost += cost
            berth_avail[b] = start + h_tilde
        stoch_costs.append(total_cost)

    mean_stoch = np.mean(stoch_costs)
    std_stoch = np.std(stoch_costs)
    print(f"  Stochastic cost: {mean_stoch:,.0f} \u00b1 {std_stoch:,.0f}")

    return best_det, mean_stoch, std_stoch, elapsed


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("GA Results for OR Spectrum Table 3")
    print("=" * 60)

    results = {}

    # Medium instance
    try:
        det, mean_s, std_s, t = run_ga_on_instance(
            'hamburg_medium_bap_vessels_with_weather.csv',
            'hamburg_medium_bap_berths.csv',
            pop_size=200, n_gen=300,
            label="Medium (20v, 8b)"
        )
        results['Medium'] = (det, mean_s, std_s, t)
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")

    # Large instance
    try:
        det, mean_s, std_s, t = run_ga_on_instance(
            'hamburg_large_bap_vessels_with_weather.csv',
            'hamburg_large_bap_berths.csv',
            pop_size=300, n_gen=500, # Changed n_gen from 400 to 500
            label="Large (40v, 16b)"
        )
        results['Large'] = (det, mean_s, std_s, t)
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Copy these values into Table 3:")
    print("=" * 60)

    # Reference FCFS values from paper
    fcfs_ref = {
        'Medium': (206557, 7550),
        'Large': (454615, 8812),
    }

    print(f"\n{'Scale':<10} {'GA Stoch Cost':<20} {'FCFS Stoch Cost':<20} {'GA Gap vs FCFS':<15}")
    print("-" * 65)
    for scale in ['Medium', 'Large']:
        if scale in results:
            _, ms, ss, _ = results[scale]
            fcfs_m, fcfs_s = fcfs_ref[scale]
            gap = (ms - fcfs_m) / fcfs_m * 100
            print(f"{scale:<10} {ms:>10,.0f} \u00b1 {ss:>5,.0f}   "
                  f"{fcfs_m:>10,} \u00b1 {fcfs_s:>5,}   {gap:>+.1f}%")

    print("\nReplace the approximate GA values in Table 3 with these actual numbers.")
import sys
import time

import pandas as pd
import shapiq

import mcs

def command_debug():
    start_time = time.time()
    game = shapiq.load_game_data(
        game_class=shapiq.GAME_NAME_TO_CLASS_MAPPING['AdultCensusLocalXAI'],
        configuration={
            'model_name': 'random_forest',
            'imputer': 'marginal'
        },
        iteration=1,
    )
    print(f"--- Loaded game in {time.time() - start_time} seconds ---")

    budget = 1000
    max_order = 3

    start_time = time.time()
    exact_computer = shapiq.ExactComputer(n_players=game.n_players, game=game)
    exact_values = exact_computer("SII", order=max_order)
    print("Exact values:", exact_values)
    print(f"--- Computed exact values in {time.time() - start_time} seconds ---")

    start_time = time.time()
    approximator = shapiq.KernelSHAPIQ(n=game.n_players, max_order=max_order, index="SII")
    approx_values = approximator(budget=budget, game=game)
    print("KernelSHAPIQ values:", approx_values)
    print(f"--- Computed KernelSHAPIQ values in {time.time() - start_time} seconds ---")

    start_time = time.time()
    approximator = shapiq.SVARMIQ(n=game.n_players, max_order=max_order, index="SII")
    approx_values = approximator(budget=budget, game=game)
    print("SVARMIQ values:", approx_values)
    print(f"--- Computed SVARMIQ values in {time.time() - start_time} seconds ---")

    start_time = time.time()
    approximator = mcs.MarginalContributionSampling(n=game.n_players, max_order=max_order, index="SII")
    approx_values = approximator(budget=budget, game=game)
    print("MCS values:", approx_values)
    print(f"--- Computed MCS values in {time.time() - start_time} seconds ---")


def benchmark_approximators(
    n: int,
    max_order: int,
    budget_steps: list[int],
    game_name: str,
    game_configuration: dict,
    game_num_instances: int = 30,
    iterations: int = 50,
):
    index = "SII"

    approximators = [
        shapiq.PermutationSamplingSII(n=n, max_order=max_order, index=index),
        # shapiq.SHAPIQ(n=n, max_order=max_order, index=index),
        # shapiq.SVARMIQ(n=n, max_order=max_order, index=index),
        # shapiq.KernelSHAPIQ(n=n, max_order=max_order, index=index),
        mcs.MarginalContributionSampling(n=n, max_order=max_order, index=index),
    ]

    df_results = pd.DataFrame(
        columns=['Game', 'Iteration', 'Approximator', 'Variant', 'Budget', 'MSE', 'Prec@10', 'Runtime']
    )
    current_df_index = 0
    
    for i in range(iterations):
        print(f"--- Iteration {i + 1}/{iterations} ---")
        game = shapiq.load_game_data(
            game_class=shapiq.GAME_NAME_TO_CLASS_MAPPING[game_name],
            configuration=game_configuration,
            iteration=(i % game_num_instances) + 1,
        )
        exact_computer = shapiq.ExactComputer(n_players=game.n_players, game=game)
        exact_values = exact_computer(index, order=max_order)

        for approximator in approximators:
            print(f"--- Approximator: {approximator.__class__.__name__} ---")
            for budget in budget_steps:
                try:
                    start_time = time.time()
                    approx_values_by_variant = approximator.approximate_variants(budget=budget, game=game)
                    elapsed_time = time.time() - start_time
                except AttributeError:
                    start_time = time.time()
                    approx_values = approximator.approximate(budget=budget, game=game)
                    elapsed_time = time.time() - start_time

                    approx_values_by_variant = {
                        '': approx_values
                    }

                for variant, approx_values in approx_values_by_variant.items():
                    if variant.startswith('player_'):
                        continue  # skip per-player variants for now

                    metrics = shapiq.benchmark.metrics.get_all_metrics(ground_truth=exact_values, estimated=approx_values)

                    df_results.loc[current_df_index] = {
                        'Game': game_name,
                        'Iteration': i + 1,
                        'Approximator': approximator.__class__.__name__,
                        'Variant': variant,
                        'Budget': budget,
                        'MSE': metrics['MSE'],
                        'Prec@10': metrics['Precision@10'],
                        'Runtime': elapsed_time,
                    }
                    current_df_index += 1

                print(f"Budget: {budget}, Runtime: {elapsed_time:.2f} seconds")
        print()

    print(df_results)


def command_benchmark():
    benchmark_approximators(
        n=14,
        max_order=2,
        budget_steps=[100, 1000, 10000],
        game_name='AdultCensusLocalXAI',
        game_configuration={
            'model_name': 'random_forest',
            'imputer': 'marginal'
        },
        game_num_instances=30,
        iterations=2,
    )




def print_help():
    print("Usage: python main.py [command] [options]")
    print("Commands:")
    print("  debug               Try simple debug run")
    print("  benchmark           Run benchmarks")
    print("  help                Show this help message")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_help()

    elif sys.argv[1] == "debug":
        command_debug()
    elif sys.argv[1] == "benchmark":
        command_benchmark()
    else:
        print_help()

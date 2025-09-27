import math
import sys
import time

import numpy as np
import pandas as pd
import shapiq

import permutationiq

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
    approximator = permutationiq.PermutationIQ(n=game.n_players, max_order=max_order, index="SII")
    approx_values = approximator(budget=budget, game=game)
    print("PermtationIQ values:", approx_values)
    print(f"--- Computed PermtationIQ values in {time.time() - start_time} seconds ---")


def benchmark_approximators(
    n: int,
    max_order: int,
    budget_steps: list[int],
    game_name: str,
    game_n_player_id: int = 0, # zero-indexed
    game_config_id: int = 1, # not zero-indexed
    game_precomputed: bool = True,
    iterations: int = 50,
    random_state: int = 1,
) -> pd.DataFrame:
    index = "SII"

    approximators = [
        shapiq.PermutationSamplingSII(n=n, max_order=max_order, index=index, random_state=random_state),
        shapiq.SHAPIQ(n=n, max_order=max_order, index=index, random_state=random_state),
        shapiq.SVARMIQ(n=n, max_order=max_order, index=index, random_state=random_state),
        shapiq.KernelSHAPIQ(n=n, max_order=max_order, index=index, random_state=random_state),
        permutationiq.PermutationIQ(n=n, max_order=max_order, index=index, random_state=random_state),
    ]

    df_results = pd.DataFrame(
        columns=['Game', 'k', 'Iteration', 'Approximator', 'Variant', 'Budget', 'MSE', 'Prec@10', 'Runtime']
    )
    current_df_index = 0

    configuration = shapiq.BENCHMARK_CONFIGURATIONS[shapiq.GAME_NAME_TO_CLASS_MAPPING[game_name]][game_n_player_id]['configurations'][game_config_id - 1]
    print("Game configuration:", configuration)

    games = [game for game in shapiq.load_games_from_configuration(
        game_class=shapiq.GAME_NAME_TO_CLASS_MAPPING[game_name],
        config_id=game_config_id,
        n_games=iterations,
        n_player_id=game_n_player_id,
        check_pre_computed=game_precomputed,
        only_pre_computed=game_precomputed,
    )]
    
    for i in range(iterations):
        print(f"--- Iteration {i + 1}/{iterations} ---")
        game = games[i % len(games)]

        start_time = time.time()
        exact_values = game.exact_values(index, order=max_order)
        print(f"Computed exact values in {time.time() - start_time} seconds")

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

                    for order in range(1, max_order + 1):
                        metrics = shapiq.benchmark.metrics.get_all_metrics(
                            ground_truth=exact_values.get_n_order(order),
                            estimated=approx_values.get_n_order(order),
                        )

                        row = {
                            'Game': game_name,
                            'k': order,
                            'Iteration': i + 1,
                            'Approximator': approximator.__class__.__name__,
                            'Variant': variant,
                            'Budget': budget,
                            'MSE': metrics['MSE'],
                            'Prec@10': metrics['Precision@10'],
                        }

                        if order == max_order:
                            row['Runtime'] = elapsed_time


                        df_results.loc[current_df_index] = row
                        current_df_index += 1

                print(f"Budget: {budget}, Runtime: {elapsed_time:.2f} seconds")
        print()

    def se(x):
        return x.std(ddof=1) / np.sqrt(x.count())

    agg_funcs = {
        'MSE': ['mean', 'std', 'min', 'max', se],
        'Prec@10': ['mean', 'std', 'min', 'max', se],
        'Runtime': ['mean', 'std', 'min', 'max', se],
    }

    # perform aggregation
    df_summary = (
        df_results
        .groupby(['Game', 'k', 'Approximator', 'Variant', 'Budget'], dropna=False)
        .agg(agg_funcs)
    )

    # flatten MultiIndex columns
    df_summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in df_summary.columns
    ]

    df_summary.reset_index(inplace=True)

    return df_summary

def benchmark_approximators_development():
    print("===============")
    print("Benchmark: approximators_development")
    print("===============")

    benchmark_approximators(
        n=14,
        max_order=2,
        budget_steps=[100, 1000, 10000],
        game_name='AdultCensusLocalXAI',
        game_n_player_id=0,
        game_config_id=2,
        iterations=2,
    ).to_csv('results/approximators_development.csv', index=False)

def benchmark_approximators_localexplanation_adultcensus():
    print("===============")
    print("Benchmark: approximators_localexplanation_adultcensus")
    print("===============")

    benchmark_approximators(
        n=14,
        max_order=4,
        budget_steps=[16, 50, 100, 250, 500, 750, 1_000, 2_000, 3_500, 5_000],
        game_name='AdultCensusLocalXAI',
        game_n_player_id=0,
        game_config_id=3,
        iterations=50,
    ).to_csv('results/approximators_localexplanation_adultcensus.csv', index=False)

def benchmark_approximators_globalexplanation_adultcensus():
    print("===============")
    print("Benchmark: approximators_globalexplanation_adultcensus")
    print("===============")

    benchmark_approximators(
        n=14,
        max_order=4,
        budget_steps=[16, 50, 100, 250, 500, 750, 1_000, 2_000, 3_500, 5_000],
        game_name='AdultCensusGlobalXAI',
        game_n_player_id=0,
        game_config_id=3,
        iterations=50,
    ).to_csv('results/approximators_globalexplanation_adultcensus.csv', index=False)

def benchmark_approximators_imageclassifier_n14():
    print("===============")
    print("Benchmark: approximators_imageclassifier_n14")
    print("===============")

    benchmark_approximators(
        n=14,
        max_order=4,
        budget_steps=[16, 50, 100, 250, 500, 750, 1_000, 2_000, 3_500, 5_000],
        game_name='ImageClassifierLocalXAI',
        game_n_player_id=0,
        game_config_id=1,
        iterations=50,
    ).to_csv('results/approximators_imageclassifier_n14.csv', index=False)

def benchmark_approximators_imageclassifier_n16():
    print("===============")
    print("Benchmark: approximators_imageclassifier_n16")
    print("===============")

    benchmark_approximators(
        n=14,
        max_order=4,
        budget_steps=[16, 50, 100, 250, 500, 750, 1_000, 2_000, 3_500, 5_000],
        game_name='ImageClassifierLocalXAI',
        game_n_player_id=2,
        game_config_id=1,
        iterations=50,
    ).to_csv('results/approximators_imageclassifier_n16.csv', index=False)

def benchmark_approximators_unsupervisedfeatureimportance_adultcensus():
    print("===============")
    print("Benchmark: approximators_unsupervisedfeatureimportance_adultcensus")
    print("===============")

    benchmark_approximators(
        n=14,
        max_order=4,
        budget_steps=[16, 50, 100, 250, 500, 750, 1_000, 2_000, 3_500, 5_000],
        game_name='AdultCensusUnsupervisedData',
        game_n_player_id=0,
        game_config_id=1,
        iterations=50,
    ).to_csv('results/approximators_unsupervisedfeatureimportance_adultcensus.csv', index=False)

def benchmark_approximators_datasetvaluation_californiahousing():
    print("===============")
    print("Benchmark: approximators_datasetvaluation_californiahousing")
    print("===============")

    benchmark_approximators(
        n=14,
        max_order=4,
        budget_steps=[16, 50, 100, 250, 500, 750, 1_000, 2_000, 3_500, 5_000],
        game_name='CaliforniaHousingDatasetValuation',
        game_n_player_id=1,
        game_config_id=1,
        iterations=50,
    ).to_csv('results/approximators_datasetvaluation_californiahousing.csv', index=False)


def benchmark_permutationiq_variants(
    min_order: int,
    max_order: int,
    budget_steps: list[int],
    game_name: str,
    game_n_player_id: int = 0,
    game_config_id: int = 1,
    game_precomputed: bool = True,
    iterations: int = 50,
    random_state: int = 1,
) -> pd.DataFrame:
    index = "SII"

    configuration = shapiq.BENCHMARK_CONFIGURATIONS[shapiq.GAME_NAME_TO_CLASS_MAPPING[game_name]][game_n_player_id]['configurations'][game_config_id - 1]
    print("Game configuration:", configuration)

    [game] = shapiq.load_games_from_configuration(
        game_class=shapiq.GAME_NAME_TO_CLASS_MAPPING[game_name],
        config_id=game_config_id,
        n_games=1,
        n_player_id=game_n_player_id,
        check_pre_computed=game_precomputed,
        only_pre_computed=game_precomputed,
    )

    start_time = time.time()
    exact_values = game.exact_values(index, order=max_order)
    print(f"Computed exact values in {time.time() - start_time} seconds")

    approximator = permutationiq.PermutationIQ(n=game.n_players, min_order=min_order, max_order=max_order, index=index, random_state=random_state)

    start_time = time.time()
    exact_variances = approximator.exact_variances(game=game, exact_values=exact_values)
    print(f"Computed exact variances in {time.time() - start_time} seconds")

    df_results = pd.DataFrame(
        columns=['Game', 'k', 'Group', 'Iteration', 'Approximator', 'Variant', 'Budget', 'Error', 'ErrorSquared']
    )
    current_df_index = 0
    
    for i in range(iterations):
        print(f"--- Iteration {i + 1}/{iterations} ---")

        for budget in budget_steps:
            try:
                start_time = time.time()
                approx_values_by_variant = approximator.approximate_variants(budget=budget, game=game, exact_variances=exact_variances)
                elapsed_time = time.time() - start_time
            except AttributeError:
                start_time = time.time()
                approx_values = approximator.approximate(budget=budget, game=game)
                elapsed_time = time.time() - start_time

                approx_values_by_variant = {
                    '': approx_values
                }

            for variant, approx_values in approx_values_by_variant.items():

                for group, group_index in approx_values.interaction_lookup.items():
                    value = approx_values.values[group_index]
                    error = exact_values.values[group_index] - value

                    if math.isnan(value):
                        continue

                    row = {
                        'Game': game_name,
                        'k': len(group),
                        'Group': group,
                        'Iteration': i + 1,
                        'Approximator': approximator.__class__.__name__,
                        'Variant': variant,
                        'Budget': budget,
                        'Error': error,
                        'ErrorSquared': error ** 2,
                    }

                    df_results.loc[current_df_index] = row
                    current_df_index += 1

            print(f"Budget: {budget}, Runtime: {elapsed_time:.2f} seconds")

        print()

    df_results.sort_values(by=['Group', 'Iteration', 'Approximator', 'Variant', 'Budget'], inplace=True)
    df_results.reset_index(inplace=True)

    def se(x):
        return x.std(ddof=1) / np.sqrt(x.count())

    agg_funcs = {
        'ErrorSquared': ['mean', 'std', 'min', 'max', se],
        'Error': ['mean', 'std', 'min', 'max', se],
    }

    # perform aggregation
    df_summary = (
        df_results
        .groupby(['Game', 'k', 'Group', 'Approximator', 'Variant', 'Budget'], dropna=False)
        .agg(agg_funcs)
    )

    # flatten MultiIndex columns
    df_summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in df_summary.columns
    ]

    df_summary.reset_index(inplace=True)

    return df_summary

def benchmark_permutationiq_variants_development():
    print("===============")
    print("Benchmark: permutationiq_variants_development")
    print("===============")

    benchmark_permutationiq_variants(
        min_order=2,
        max_order=2,
        budget_steps=[100, 1000, 10000],
        game_name='AdultCensusLocalXAI',
        game_n_player_id=0,
        game_config_id=2,
        iterations=2,
    ).to_csv('results/permutationiq_variants_development.csv', index=False)

def benchmark_permutationiq_variants_localexplanation_adultcensus():
    print("===============")
    print("Benchmark: permutationiq_variants_localexplanation_adultcensus")
    print("===============")

    benchmark_permutationiq_variants(
        min_order=2,
        max_order=3,
        budget_steps=[16, 50, 100, 250, 500, 750, 1_000, 2_000, 3_500, 5_000, 7_500, 10_000],
        game_name='AdultCensusLocalXAI',
        game_n_player_id=0,
        game_config_id=3,
        iterations=50,
    ).to_csv('results/permutationiq_variants_localexplanation_adultcensus.csv', index=False)

def benchmark_permutationiq_variants_globalexplanation_adultcensus():
    print("===============")
    print("Benchmark: permutationiq_variants_globalexplanation_adultcensus")
    print("===============")

    benchmark_permutationiq_variants(
        min_order=2,
        max_order=3,
        budget_steps=[16, 50, 100, 250, 500, 750, 1_000, 2_000, 3_500, 5_000, 7_500, 10_000],
        game_name='AdultCensusGlobalXAI',
        game_n_player_id=0,
        game_config_id=3,
        iterations=50,
    ).to_csv('results/permutationiq_variants_globalexplanation_adultcensus.csv', index=False)

def benchmark_permutationiq_variants_soum():
    print("===============")
    print("Benchmark: permutationiq_variants_soum")
    print("===============")

    benchmark_permutationiq_variants(
        min_order=2,
        max_order=3,
        budget_steps=[16, 50, 100, 250, 500, 750, 1_000, 2_000, 3_500, 5_000, 7_500, 10_000],
        game_name='SOUM',
        game_n_player_id=0,
        game_config_id=4,
        game_precomputed=False,
        iterations=50,
    ).to_csv('results/permutationiq_variants_soug.csv', index=False)


def command_benchmark(config: str):
    if config == "all" or config == "approximators_development":
        benchmark_approximators_development()
    if config == "all" or config == "approximators_localexplanation_adultcensus":
        benchmark_approximators_localexplanation_adultcensus()
    if config == "all" or config == "approximators_globalexplanation_adultcensus":
        benchmark_approximators_globalexplanation_adultcensus()
    if config == "all" or config == "approximators_imageclassifier_n14":
        benchmark_approximators_imageclassifier_n14()
    if config == "all" or config == "approximators_imageclassifier_n16":
        benchmark_approximators_imageclassifier_n16()
    if config == "all" or config == "approximators_unsupervisedfeatureimportance_adultcensus":
        benchmark_approximators_unsupervisedfeatureimportance_adultcensus()
    if config == "all" or config == "approximators_datasetvaluation_californiahousing":
        benchmark_approximators_datasetvaluation_californiahousing()

    if config == "all" or config == "permutationiq_variants_development":
        benchmark_permutationiq_variants_development()
    if config == "all" or config == "permutationiq_variants_localexplanation_adultcensus":
        benchmark_permutationiq_variants_localexplanation_adultcensus()
    if config == "all" or config == "permutationiq_variants_globalexplanation_adultcensus":
        benchmark_permutationiq_variants_globalexplanation_adultcensus()
    if config == "all" or config == "permutationiq_variants_soum":
        benchmark_permutationiq_variants_soum()




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
        command_benchmark(sys.argv[2] if len(sys.argv) >= 3 else "all")
    else:
        print_help()

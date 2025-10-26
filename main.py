import math
import sys
import time
import signal
from contextlib import contextmanager

import numpy as np
import pandas as pd
import shapiq

import permutationiq

class TimeoutException(Exception): pass

# copied from https://stackoverflow.com/a/601168/2796524
# timeout must be given in full seconds; timeout of zero or less raises timeout immediately
@contextmanager
def time_limit(timeout_secs: int):
    if timeout_secs <= 0:
        raise TimeoutException("Timed out!")

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(timeout_secs)
    try:
        yield
    finally:
        signal.alarm(0)


def command_debug():
    start_time = time.time()
    # game = shapiq.load_game_data(
    #     game_class=shapiq.GAME_NAME_TO_CLASS_MAPPING['AdultCensusLocalXAI'],
    #     configuration={
    #         'model_name': 'random_forest',
    #         'imputer': 'marginal'
    #     },
    #     iteration=1,
    # )
    game = shapiq.games.benchmark.SOUM(
        n = 10,
        n_basis_games = 150,
        min_interaction_size = 1,
        max_interaction_size = 2,
        random_state = 1,
    )
    print(f"--- Loaded game in {time.time() - start_time} seconds ---")

    budget = 10000
    max_order = 2

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
    print("PermutationIQ values:", approx_values)
    print(f"--- Computed PermutationIQ values in {time.time() - start_time} seconds ---")

    start_time = time.time()
    approximator = permutationiq.PermutationIQStratified(n=game.n_players, max_order=max_order, index="SII")
    approx_values = approximator(budget=budget, game=game)
    print("PermutationIQ+Strat values:", approx_values)
    print(f"--- Computed PermutationIQ+Strat values in {time.time() - start_time} seconds ---")


def benchmark_approximators(
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

    n = shapiq.BENCHMARK_CONFIGURATIONS[shapiq.GAME_NAME_TO_CLASS_MAPPING[game_name]][game_n_player_id]['n_players']
    print("n =", n)

    configuration = shapiq.BENCHMARK_CONFIGURATIONS[shapiq.GAME_NAME_TO_CLASS_MAPPING[game_name]][game_n_player_id]['configurations'][game_config_id - 1]
    print("Game configuration:", configuration)

    approximators = [
        shapiq.PermutationSamplingSII(n=n, max_order=max_order, index=index, random_state=random_state),
        shapiq.SHAPIQ(n=n, max_order=max_order, index=index, random_state=random_state),
        shapiq.SVARMIQ(n=n, max_order=max_order, index=index, random_state=random_state),
        shapiq.KernelSHAPIQ(n=n, max_order=max_order, index=index, random_state=random_state),
        permutationiq.PermutationIQ(n=n, max_order=max_order, index=index, random_state=random_state),
        permutationiq.PermutationIQStratified(n=n, max_order=max_order, index=index, random_state=random_state),
    ]

    df_results = pd.DataFrame(
        columns=['Game', 'k', 'n', 'Iteration', 'Approximator', 'Variant', 'Budget', 'MSE', 'Prec10', 'Runtime']
    )
    current_df_index = 0

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
                            'n': n,
                            'k': order,
                            'Iteration': i + 1,
                            'Approximator': approximator.__class__.__name__,
                            'Variant': variant,
                            'Budget': budget,
                            'MSE': metrics['MSE'],
                            'Prec10': metrics['Precision@10'],
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
        'Prec10': ['mean', 'std', 'min', 'max', se],
        'Runtime': ['mean', 'std', 'min', 'max', se],
    }

    # perform aggregation
    df_summary = (
        df_results
        .groupby(['Game', 'n', 'k', 'Approximator', 'Variant', 'Budget'], dropna=False)
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
        max_order=4,
        budget_steps=[16, 50, 100, 250, 500, 750, 1_000, 2_000, 3_500, 5_000],
        game_name='CaliforniaHousingDatasetValuation',
        game_n_player_id=1,
        game_config_id=1,
        iterations=50,
    ).to_csv('results/approximators_datasetvaluation_californiahousing.csv', index=False)


def _benchmark_runtime_soum_varying_n(
    max_order: int,
    budget: int,
    n_steps=[10, 15, 30, 50, 75, 100],
    n_basis_games: int = 150,
    iterations: int = 50,
    random_state: int = 1,
    time_limit_secs: int = 3600,
) -> pd.DataFrame:
    index = "SII"

    shapiq.games.benchmark.SOUM

    df_results = pd.DataFrame(
        columns=['Game', 'k', 'n', 'Iteration', 'Approximator', 'Variant', 'Budget', 'MSE', 'Prec10', 'Runtime']
    )
    current_df_index = 0

    timeouted_approximators = set()

    for n in n_steps:
        print(f"--- n = {n} ---")

        approximators = {
            shapiq.PermutationSamplingSII: shapiq.PermutationSamplingSII(n=n, max_order=max_order, index=index, random_state=random_state),
            shapiq.SHAPIQ: shapiq.SHAPIQ(n=n, max_order=max_order, index=index, random_state=random_state),
            shapiq.SVARMIQ: shapiq.SVARMIQ(n=n, max_order=max_order, index=index, random_state=random_state),
            shapiq.KernelSHAPIQ: shapiq.KernelSHAPIQ(n=n, max_order=max_order, index=index, random_state=random_state),
            permutationiq.PermutationIQ: permutationiq.PermutationIQ(n=n, max_order=max_order, index=index, random_state=random_state),
            permutationiq.PermutationIQStratified: permutationiq.PermutationIQStratified(n=n, max_order=max_order, index=index, random_state=random_state),
        }

        for i in range(iterations):
            print(f"--- Iteration {i + 1}/{iterations} ---")

            game = shapiq.games.benchmark.SOUM(
                n = n,
                n_basis_games = n_basis_games,
                min_interaction_size = 1,
                max_interaction_size = n,
                random_state = random_state + n * 2 + i * 3, # since 2 and 3 are primes, this should give different seeds for different n and i
            )

            start_time = time.time()
            exact_values = game.exact_values(index, order=max_order)
            print(f"Computed exact values in {time.time() - start_time} seconds")

            for (approximator_class, approximator) in approximators.items():
                if approximator_class in timeouted_approximators:
                    print(f"--- Approximator: {approximator_class.__name__} skipped due to previous timeout ---")
                    continue

                print(f"--- Approximator: {approximator_class.__name__} ---")

                try:
                    start_time = time.time()
                    with time_limit(time_limit_secs):
                        approx_values = approximator.approximate(budget=budget, game=game)
                    elapsed_time = time.time() - start_time

                except TimeoutException:
                    elapsed_time = time.time() - start_time
                    print(f"Timed out after {elapsed_time} seconds, approximator {approximator_class.__name__} will be skipped for further iterations.")
                    timeouted_approximators.add(approximator_class)
                    continue

                metrics = shapiq.benchmark.metrics.get_all_metrics(
                    ground_truth=exact_values.get_n_order(max_order),
                    estimated=approx_values.get_n_order(max_order),
                )

                row = {
                    'Game': 'SOUM',
                    'n': n,
                    'k': max_order,
                    'Iteration': i + 1,
                    'Approximator': approximator.__class__.__name__,
                    'Budget': budget,
                    'MSE': metrics['MSE'],
                    'Prec10': metrics['Precision@10'],
                    'Runtime': elapsed_time,
                }

                df_results.loc[current_df_index] = row
                current_df_index += 1

                print(f"Runtime: {elapsed_time:.2f} seconds")
        
            print()

    def se(x):
        return x.std(ddof=1) / np.sqrt(x.count())

    agg_funcs = {
        'MSE': ['mean', 'std', 'min', 'max', se],
        'Prec10': ['mean', 'std', 'min', 'max', se],
        'Runtime': ['mean', 'std', 'min', 'max', se],
    }

    # perform aggregation
    df_summary = (
        df_results
        .groupby(['Game', 'n', 'k', 'Approximator', 'Budget'], dropna=False)
        .agg(agg_funcs)
    )

    # flatten MultiIndex columns
    df_summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in df_summary.columns
    ]

    df_summary['Iterations'] = df_results.groupby(['Game', 'n', 'k', 'Approximator', 'Budget'], dropna=False).size()
    df_summary.reset_index(inplace=True)

    return df_summary

def benchmark_runtime_soum_varying_n():
    print("===============")
    print("Benchmark: runtime_soum_varying_n")
    print("===============")

    _benchmark_runtime_soum_varying_n(
        max_order=2,
        budget=10_000,
        n_steps=[15, 30, 50, 70, 85, 100],
        iterations=30,
    ).to_csv('results/runtime_soum_varying_n.csv', index=False)


def _benchmark_runtime_varying_k(
    max_order_steps: list[int],
    budget: int,
    game_name: str,
    game_n_player_id: int = 0, # zero-indexed
    game_config_id: int = 1, # not zero-indexed
    game_precomputed: bool = True,
    iterations: int = 50,
    random_state: int = 1,
    time_limit_secs: int = 3600,
) -> pd.DataFrame:
    index = "SII"

    n = shapiq.BENCHMARK_CONFIGURATIONS[shapiq.GAME_NAME_TO_CLASS_MAPPING[game_name]][game_n_player_id]['n_players']
    print("n =", n)

    configuration = shapiq.BENCHMARK_CONFIGURATIONS[shapiq.GAME_NAME_TO_CLASS_MAPPING[game_name]][game_n_player_id]['configurations'][game_config_id - 1]
    print("Game configuration:", configuration)

    df_results = pd.DataFrame(
        columns=['Game', 'k', 'n', 'Iteration', 'Approximator', 'Variant', 'Budget', 'MSE', 'Prec10', 'Runtime']
    )
    current_df_index = 0

    timeouted_approximators = set()

    games = [game for game in shapiq.load_games_from_configuration(
        game_class=shapiq.GAME_NAME_TO_CLASS_MAPPING[game_name],
        config_id=game_config_id,
        n_games=iterations,
        n_player_id=game_n_player_id,
        check_pre_computed=game_precomputed,
        only_pre_computed=game_precomputed,
    )]

    for max_order in max_order_steps:
        print(f"--- k = {max_order} ---")

        approximators = {
            shapiq.PermutationSamplingSII: shapiq.PermutationSamplingSII(n=n, max_order=max_order, index=index, random_state=random_state),
            shapiq.SHAPIQ: shapiq.SHAPIQ(n=n, max_order=max_order, index=index, random_state=random_state),
            shapiq.SVARMIQ: shapiq.SVARMIQ(n=n, max_order=max_order, index=index, random_state=random_state),
            shapiq.KernelSHAPIQ: shapiq.KernelSHAPIQ(n=n, max_order=max_order, index=index, random_state=random_state),
            permutationiq.PermutationIQ: permutationiq.PermutationIQ(n=n, max_order=max_order, index=index, random_state=random_state),
            permutationiq.PermutationIQStratified: permutationiq.PermutationIQStratified(n=n, max_order=max_order, index=index, random_state=random_state),
        }

        for i in range(iterations):
            print(f"--- Iteration {i + 1}/{iterations} ---")
            game = games[i % len(games)]

            start_time = time.time()
            exact_values = game.exact_values(index, order=max_order)
            print(f"Computed exact values in {time.time() - start_time} seconds")

            for (approximator_class, approximator) in approximators.items():
                if approximator_class in timeouted_approximators:
                    print(f"--- Approximator: {approximator_class.__name__} skipped due to previous timeout ---")
                    continue

                print(f"--- Approximator: {approximator_class.__name__} ---")

                try:
                    start_time = time.time()
                    with time_limit(time_limit_secs):
                        approx_values = approximator.approximate(budget=budget, game=game)
                    elapsed_time = time.time() - start_time

                except TimeoutException:
                    elapsed_time = time.time() - start_time
                    print(f"Timed out after {elapsed_time} seconds, approximator {approximator_class.__name__} will be skipped for further iterations.")
                    timeouted_approximators.add(approximator_class)
                    continue

                metrics = shapiq.benchmark.metrics.get_all_metrics(
                    ground_truth=exact_values.get_n_order(max_order),
                    estimated=approx_values.get_n_order(max_order),
                )

                row = {
                    'Game': 'SOUM',
                    'n': n,
                    'k': max_order,
                    'Iteration': i + 1,
                    'Approximator': approximator.__class__.__name__,
                    'Budget': budget,
                    'MSE': metrics['MSE'],
                    'Prec10': metrics['Precision@10'],
                    'Runtime': elapsed_time,
                }

                df_results.loc[current_df_index] = row
                current_df_index += 1

                print(f"Runtime: {elapsed_time:.2f} seconds")
        
            print()

    def se(x):
        return x.std(ddof=1) / np.sqrt(x.count())

    agg_funcs = {
        'MSE': ['mean', 'std', 'min', 'max', se],
        'Prec10': ['mean', 'std', 'min', 'max', se],
        'Runtime': ['mean', 'std', 'min', 'max', se],
    }

    # perform aggregation
    df_summary = (
        df_results
        .groupby(['Game', 'n', 'k', 'Approximator', 'Budget'], dropna=False)
        .agg(agg_funcs)
    )

    # flatten MultiIndex columns
    df_summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in df_summary.columns
    ]

    df_summary['Iterations'] = df_results.groupby(['Game', 'n', 'k', 'Approximator', 'Budget'], dropna=False).size()
    df_summary.reset_index(inplace=True)

    return df_summary

def benchmark_runtime_soum_varying_k():
    print("===============")
    print("Benchmark: runtime_soum_varying_k")
    print("===============")

    _benchmark_runtime_varying_k(
        max_order_steps=[1, 2, 3, 4, 5],
        budget=10_000,
        game_name='SOUM',
        game_n_player_id=0,
        game_config_id=4,
        game_precomputed=False,
        iterations=30,
    ).to_csv('results/runtime_soum_varying_k.csv', index=False)


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
    ).to_csv('results/permutationiq_variants_soum.csv', index=False)


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

    if config == "all" or config == "runtime_soum_varying_n":
        benchmark_runtime_soum_varying_n()

    if config == "all" or config == "runtime_soum_varying_k":
        benchmark_runtime_soum_varying_k()




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

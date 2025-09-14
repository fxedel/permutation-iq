import time

import shapiq

import mcs

def main():
    start_time = time.time()
    game = shapiq.load_game_data(
        game_class=shapiq.GAME_NAME_TO_CLASS_MAPPING['AdultCensusLocalXAI'],
        configuration={
            'model_name': 'random_forest',
            'imputer': 'marginal'
        },
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


if __name__ == "__main__":
    main()

import math
from typing import Literal, Callable

import numpy as np
import shapiq
from shapiq.interaction_values import InteractionValues

ValidIndices = Literal["SII"]

class OnlineMeanVariance:
    '''
    Estimator for mean and variance which can be updated online.
    Uses Welford's online algorithm.
    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    '''

    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None
        self.variance = None

    def __repr__(self):
        return f'OnlineVariance(n={self.n}, mean={self.mean}, variance={self.variance})'

    def update(self, x: float):
        self.n += 1

        if self.mean is None:
            self.mean = x
            self.m2 = 0
            return

        delta = x - self.mean
        self.mean = ((self.mean * (self.n - 1)) + x) / self.n
        self.m2 += delta * (x - self.mean)

        if self.n > 1:
            self.variance = self.m2 / (self.n - 1)

class MarginalContributionSampling(shapiq.approximator.Approximator):

    def __init__(
        self,
        n: int,
        max_order: int = 2,
        index: ValidIndices = "SII",
        *,
        top_order: bool = False,
        random_state: int | None = None,
    ) -> None:
        """Initialize the SVARMIQ approximator.

        Args:
            n: The number of players.

            max_order: The interaction order of the approximation. Defaults to ``2``.

            index: The interaction index to be used. Choose from ``['k-SII', 'SII']``. Defaults to
                ``'k-SII'``.

            top_order: If ``True``, only the top-order interactions are estimated. Defaults to ``False``.

            random_state: The random state of the estimator. Defaults to ``None``.

        """
        super().__init__(
            n,
            max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
        )

        self.iteration_cost = n

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
    ) -> InteractionValues:
        return self.approximate_variants(budget, game)['mean']

    def approximate_variants(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
    ) -> dict[str, InteractionValues]:
        players = list(range(self.n))

        player_group_estimates: dict[int, dict[tuple[int], OnlineMeanVariance]] = {
            player: {} for player in players
        }
        for group in self._interaction_lookup.keys():
            for player in group:
                player_group_estimates[player][group] = OnlineMeanVariance()

        used_budget = 0

        empty_value = game(np.zeros(self.n, dtype=bool))[0]
        used_budget += 1

        while used_budget < budget:
            perm = players.copy()
            self._shuffle(perm)

            coalition = np.zeros(self.n, dtype=bool)
            prev_value = empty_value

            for (i, player) in enumerate(perm):
                coalition[player] = True
                value = game(coalition)[0]
                used_budget += 1

                marginal_contribution = value - prev_value
                prev_value = value

                sample_probability = (1 / self.n) * (1 / math.comb(self.n - 1, i))

                for group, group_estimate in player_group_estimates[player].items():
                    k = len(group)

                    group_subset_len = 0
                    for x in group:
                        if coalition[x] and x != player:
                            group_subset_len += 1

                    basis_coalition_len = i - group_subset_len

                    sign = 1 if (k - group_subset_len - 1) % 2 == 0 else -1
                    weight = shapley_weight(n = self.n, k = k, l = basis_coalition_len)

                    group_estimate.update(sign * weight/sample_probability * marginal_contribution)

                if used_budget >= budget:
                    break

        group_player_estimates = {
            group: {
                player: player_group_estimates[player][group] for player in group
            } for group in self._interaction_lookup.keys() if group != ()
        }

        variant_group_interactions = {
            'mean': {},
            'inverse_variance_weighting': {},
            # 'inverse_variance_weighting_optimal': {}
        }
        for player in players:
            variant_group_interactions[f'player_{player}'] = {}

        # print('Estimated variances:')

        for group, player_estimates in group_player_estimates.items():
            for player in group:
                variant_group_interactions[f'player_{player}'][group] = player_estimates[player].mean

            estimates_with_mean = {player: estimate for player, estimate in player_estimates.items() if estimate.mean is not None}

            estimates_sum = sum([estimate.mean for estimate in estimates_with_mean.values()])
            estimates_count = len(estimates_with_mean)
            variant_group_interactions['mean'][group] = estimates_sum / estimates_count if estimates_count > 0 else None

            estimates_with_variance = {player: estimate for player, estimate in estimates_with_mean.items() if estimate.variance is not None}

            # for player, estimate in estimates_with_variance.items():
            #     print(f'> Group {group}, player {player}: {estimate.variance:.6f} (n = {estimate.n})')

            zero_variance_estimates = [estimate for estimate in estimates_with_variance.values() if estimate.variance == 0]
            if len(zero_variance_estimates) > 0:
                variant_group_interactions['inverse_variance_weighting'][group] = sum([estimate.mean for estimate in zero_variance_estimates]) / len(zero_variance_estimates)
            elif len(estimates_with_variance) > 0:
                variant_group_interactions['inverse_variance_weighting'][group] = sum([estimate.mean / (estimate.variance / estimate.n) for estimate in estimates_with_variance.values()]) / sum([1 / (estimate.variance / estimate.n) for estimate in estimates_with_variance.values()])
            else:
                variant_group_interactions['inverse_variance_weighting'][group] = None

            # if exact_variances is not None:
            #     zero_variance_estimates = [estimate for player, estimate in estimates_with_mean.items() if exact_variances[group][player] == 0]
            #     if len(zero_variance_estimates) > 0:
            #         variant_group_interactions['inverse_variance_weighting_optimal'][group] = sum([estimate.mean for estimate in zero_variance_estimates]) / len(zero_variance_estimates)
            #     elif len(estimates_with_mean) > 0:
            #         variant_group_interactions['inverse_variance_weighting_optimal'][group] = sum([estimate.mean / (exact_variances[group][player] / estimate.n) for player, estimate in estimates_with_mean.items()]) / sum([1 / (exact_variances[group][player] / estimate.n) for player, estimate in estimates_with_mean.items()])
            #     else:
            #         variant_group_interactions['inverse_variance_weighting_optimal'][group] = None

        variant_interaction_values = {}

        for variant, group_interactions in variant_group_interactions.items():
            result = self._init_result()

            for group, group_index in self._interaction_lookup.items():
                if group == ():
                    result[group_index] = empty_value
                    continue

                result[group_index] = group_interactions.get(group)

            variant_interaction_values[variant] = InteractionValues(
                values=result,
                index=self.approximation_index,
                max_order=self.max_order,
                n_players=self.n,
                min_order=self.min_order,
                baseline_value=empty_value,
                interaction_lookup=self._interaction_lookup,
                estimated=True,
                estimation_budget=used_budget,
            )
        
        return variant_interaction_values


    def _shuffle(self, arr: list) -> list:
        """in-place Fisher-Yates shuffle, see https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle"""

        for i in range(len(arr)-1):
            j = self._rng.integers(i, len(arr))
            self._swap(arr, i, j)

        return arr
    
    def _swap(self, arr: list, i: int, j: int) -> list:
        arr[i], arr[j] = arr[j], arr[i]
        return arr

def shapley_weight(n: int, k: int, l: int) -> float:
    return 1 / (n - k + 1) / math.comb(n - k, l)

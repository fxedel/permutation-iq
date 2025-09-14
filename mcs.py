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


        result = self._init_result()

        for group, group_index in self._interaction_lookup.items():
            if group == ():
                result[group_index] = empty_value
                continue

            group_estimate = 0.0
            for player in group:
                group_estimate += player_group_estimates[player][group].mean or 0.0

            result[group_index] = group_estimate / len(group)

        return InteractionValues(
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

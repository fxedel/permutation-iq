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

class PermutationIQ(shapiq.approximator.Approximator):

    def __init__(
        self,
        n: int,
        min_order: int = 0,
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
            n=n,
            min_order=min_order,
            max_order=max_order,
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
        exact_variances: dict[tuple, dict[int, float]] | None = None,
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

        iterations = (budget-used_budget) // self.n

        coalitions_base = np.tri(self.n, self.n, k=0, dtype=bool)
        coalitions = np.empty((iterations * self.n, self.n), dtype=bool)
        permutations = []

        for j in range(iterations):
            perm = players.copy()
            self._rng.shuffle(perm)
            permutations.append(perm)

            coalitions[j*self.n:(j+1)*self.n] = coalitions_base[:, np.argsort(perm)]

        coalition_values = game(coalitions)
        used_budget += len(coalition_values)
        if used_budget > budget:
            raise RuntimeError('Exceeded budget!') # this shouldn't happen

        for j in range(iterations):
            perm = permutations[j]
            iteration_coalitions = coalitions[j*self.n:(j+1)*self.n]
            iteration_coalition_values = coalition_values[j*self.n:(j+1)*self.n]

            prev_value = empty_value

            for (i, (player, coalition, coalition_value)) in enumerate(zip(perm, iteration_coalitions, iteration_coalition_values)):
                marginal_contribution = coalition_value - prev_value
                prev_value = coalition_value

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

        return aggregate_group_player_estimates(
            players=players,
            empty_value=empty_value,
            used_budget=used_budget,
            group_player_estimates=group_player_estimates,
            init_result=self._init_result,
            interaction_lookup=self._interaction_lookup,
            approximation_index=self.approximation_index,
            max_order=self.max_order,
            n=self.n,
            min_order=self.min_order,
            exact_variances=exact_variances,
        )

    def exact_variances(
        self,
        game: Callable[[np.ndarray], np.ndarray],
        exact_values: InteractionValues,
    ) -> dict[tuple, dict[int, float]]:
        players = list(range(self.n))

        player_group_variances: dict[int, dict[tuple[int, ...], float]] = {player: {} for player in players}
        for group in self._interaction_lookup.keys():
            for player in group:
                player_group_variances[player][group] = 0.0

        for player in players:
            for bitvector in range(2**self.n):
                coalition = np.zeros(self.n, dtype=bool)
                for i in players:
                    if bitvector & (1 << i):
                        coalition[i] = True

                if coalition[player]:
                    continue

                coalition_value = game(coalition)[0]
                # print(coalition, coalition_value)
                coalition_len = len([x for x in coalition if x])

                coalition[player] = True
                marginal_contribution = game([coalition])[0] - coalition_value
                # print(coalition, game([coalition]))
                # print(marginal_contribution)

                sample_probability = (1 / self.n) * (1 / math.comb(self.n - 1, coalition_len))

                for group in player_group_variances[player]:
                    group_subset_len = 0
                    for x in group:
                        if x != player and coalition[x]:
                            group_subset_len += 1

                    basis_coalition_len = coalition_len - group_subset_len

                    # print(coalition, coalition_len, group, group_subset_len, basis_coalition_len, self.n, len(group))

                    sign = 1 if (len(group) - group_subset_len - 1) % 2 == 0 else -1
                    weight = shapley_weight(n = self.n, k = len(group), l = basis_coalition_len)

                    # print(group, player_group_variances[player][group], sample_probability, sign, weight, marginal_contribution, exact_values[group])

                    player_group_variances[player][group] += sample_probability * (sign * weight/sample_probability * marginal_contribution - exact_values[group])**2


        group_player_variances = {
            group: {
                player: player_group_variances[player][group] for player in group
            } for group in self._interaction_lookup.keys() if group != ()
        }

        return group_player_variances

class PermutationIQStratified(shapiq.approximator.Approximator):

    def __init__(
        self,
        n: int,
        min_order: int = 0,
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
            n=n,
            min_order=min_order,
            max_order=max_order,
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
        exact_variances: dict[tuple, dict[int, float]] | None = None,
    ) -> dict[str, InteractionValues]:
        players = list(range(self.n))
        strata = list(range(self.n))

        player_group_stratum_estimates: dict[int, dict[tuple[int, ...], list[OnlineMeanVariance]]] = {
            player: {} for player in players
        }
        for group in self._interaction_lookup.keys():
            for player in group:
                player_group_stratum_estimates[player][group] = [OnlineMeanVariance() for _ in strata]

        used_budget = 0

        empty_value = game(np.zeros(self.n, dtype=bool))[0]
        used_budget += 1

        iterations = (budget-used_budget) // self.n

        if iterations > 0:
            coalitions_base = np.tri(self.n, self.n, k=0, dtype=bool)
            coalitions = np.empty((iterations * self.n, self.n), dtype=bool)
            permutations = []

            for j in range(iterations):
                perm = players.copy()
                self._rng.shuffle(perm)
                permutations.append(perm)

                coalitions[j*self.n:(j+1)*self.n] = coalitions_base[:, np.argsort(perm)]

            coalition_values = game(coalitions)
            used_budget += len(coalition_values)
            if used_budget > budget:
                raise RuntimeError('Exceeded budget!') # this shouldn't happen

            for j in range(iterations):
                perm = permutations[j]
                iteration_coalitions = coalitions[j*self.n:(j+1)*self.n]
                iteration_coalition_values = coalition_values[j*self.n:(j+1)*self.n]

                prev_value = empty_value

                for (i, (player, coalition, coalition_value)) in enumerate(zip(perm, iteration_coalitions, iteration_coalition_values)):
                    marginal_contribution = coalition_value - prev_value
                    prev_value = coalition_value

                    for group, group_stratum_estimates in player_group_stratum_estimates[player].items():
                        k = len(group)

                        group_subset_len = 0
                        for x in group:
                            if coalition[x] and x != player:
                                group_subset_len += 1

                        basis_coalition_len = i - group_subset_len

                        sign = 1 if (k - group_subset_len - 1) % 2 == 0 else -1
                        weight = shapley_weight(n = self.n, k = k, l = basis_coalition_len)

                        group_stratum_estimates[i].update(sign * weight * marginal_contribution)

                    if used_budget >= budget:
                        break

        group_player_estimates = {}
        for group in self._interaction_lookup.keys():
            if group == ():
                continue

            group_player_estimates[group] = {}

            for player in group:
                stratum_estimates = player_group_stratum_estimates[player][group]

                stratum_estimates_with_mean = {
                    stratum: stratum_estimate
                    for stratum, stratum_estimate in enumerate(stratum_estimates)
                    if stratum_estimate.mean != None
                }
                stratum_estimates_with_variance = {
                    stratum: stratum_estimate
                    for stratum, stratum_estimate in enumerate(stratum_estimates)
                    if stratum_estimate.variance != None
                }

                interaction_estimate = OnlineMeanVariance()

                if len(stratum_estimates_with_mean) > 0:
                    # cheat a little bit: interaction_estimate.n is actually number of strata,
                    # but we want the number of underlying samples
                    interaction_estimate.n = sum([estimate.n for estimate in stratum_estimates_with_mean.values()])
                    interaction_estimate.mean = sum([
                        estimate.mean * math.comb(self.n - 1, stratum) * len(strata) # scale with inverse sample_probability
                        for (stratum, estimate) in stratum_estimates_with_mean.items()
                    ]) / len(stratum_estimates_with_mean)

                    if len(stratum_estimates_with_variance) > 0:
                        interaction_estimate.variance = sum([
                            estimate.variance * (math.comb(self.n - 1, stratum) * len(strata))**2 # scale with inverse sample_probability
                            for (stratum, estimate) in stratum_estimates_with_variance.items()
                        ]) / (len(stratum_estimates_with_variance)**2)

                group_player_estimates[group][player] = interaction_estimate

        return aggregate_group_player_estimates(
            players=players,
            empty_value=empty_value,
            used_budget=used_budget,
            group_player_estimates=group_player_estimates,
            init_result=self._init_result,
            interaction_lookup=self._interaction_lookup,
            approximation_index=self.approximation_index,
            max_order=self.max_order,
            n=self.n,
            min_order=self.min_order,
            exact_variances=exact_variances,
        )


def aggregate_group_player_estimates(
    players,
    empty_value,
    used_budget,
    group_player_estimates,
    init_result,
    interaction_lookup,
    approximation_index,
    max_order,
    n,
    min_order,
    exact_variances = None,
):
    variant_group_interactions = {
        'mean': {},
        'inverse_variance_weighting': {},
        'inverse_variance_weighting_optimal': {}
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

        if exact_variances is not None:
            zero_variance_estimates = [estimate for player, estimate in estimates_with_mean.items() if exact_variances[group][player] == 0]
            if len(zero_variance_estimates) > 0:
                variant_group_interactions['inverse_variance_weighting_optimal'][group] = sum([estimate.mean for estimate in zero_variance_estimates]) / len(zero_variance_estimates)
            elif len(estimates_with_mean) > 0:
                variant_group_interactions['inverse_variance_weighting_optimal'][group] = sum([estimate.mean / (exact_variances[group][player] / estimate.n) for player, estimate in estimates_with_mean.items()]) / sum([1 / (exact_variances[group][player] / estimate.n) for player, estimate in estimates_with_mean.items()])
            else:
                variant_group_interactions['inverse_variance_weighting_optimal'][group] = None

    variant_interaction_values = {}

    for variant, group_interactions in variant_group_interactions.items():
        result = init_result()

        for group, group_index in interaction_lookup.items():
            if group == ():
                result[group_index] = empty_value
                continue

            result[group_index] = group_interactions.get(group)

        variant_interaction_values[variant] = InteractionValues(
            values=result,
            index=approximation_index,
            max_order=max_order,
            n_players=n,
            min_order=min_order,
            baseline_value=empty_value,
            interaction_lookup=interaction_lookup,
            estimated=True,
            estimation_budget=used_budget,
        )

    return variant_interaction_values

def shapley_weight(n: int, k: int, l: int) -> float:
    return 1 / (n - k + 1) / math.comb(n - k, l)

class Subsets:
    '''
    Iterator over all subsets of a list
    '''

    def __init__(self, set: list):
        self.set = set
        self.current = 0
        self.limit = 2**(len(set))

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration

        subset = [item for (i, item) in enumerate(self.set) if self.current & (1 << i)]

        self.current += 1
        return subset

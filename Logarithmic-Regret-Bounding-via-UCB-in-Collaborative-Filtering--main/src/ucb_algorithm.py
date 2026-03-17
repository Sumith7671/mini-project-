import math

import numpy as np


class UCB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self, t):
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            bonus = math.sqrt((2 * math.log(t)) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus

        return int(np.argmax(ucb_values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward

        self.values[chosen_arm] = new_value


#!/usr/bin/python3
import numpy as np
from itertools import product

from learning.model_based import Problem
from learning.model_based import policy_iteration
from learning.model_based import value_iteration


class GridProblem(Problem):

    def __init__(self, side=4, final=0):
        self.side = side
        self.states = range(side * side)

        self.final = final

        self.actions = range(4)
        self.actions_repr = np.array(['r', 'l', 'u', 'd'])

        self.probs = np.zeros(
            (len(self.states), len(self.states), len(self.actions)))
        self.rewards = np.zeros(
            (len(self.states), len(self.states), len(self.actions)))

        for (s, next_s, action) in product(self.states, self.states, self.actions):
            self.probs[(s, next_s, action)] = self._p(s, next_s, action)
            self.rewards[(s, next_s, action)] = self._r(s, next_s, action)

    def rlud(self, s):
        i, j = s / self.side, s % self.side

        states = [s] * 4
        if j + 1 < self.side:
            states[0] = s + 1
        if j - 1 >= 0:
            states[1] = s - 1
        if i - 1 >= 0:
            states[2] = s - self.side
        if i + 1 < self.side:
            states[3] = s + self.side

        return states

    def p(self, s, next_s, action):
        return self.probs[s, next_s, action]

    def r(self, s, next_s, action):
        return self.rewards[s, next_s, action]

    def _p(self, s, next_s, action):
        if s == self.final:
            if next_s == s:
                return 1
            else:
                return 0
        elif self.rlud(s)[action] == next_s:
            return 1

        return 0

    def _r(self, s, next_s, action):
        if s == self.final:
            return 0

        return -1

    def a(self, s):
        return self.actions

    def print_policy(self, policy):
        P = np.array(policy).reshape(self.side, self.side)
        print(self.actions_repr[P])

    def print_values(self, values):
        V = np.array(values).reshape(self.side, self.side)
        print(V)


def main():
    np.set_printoptions(precision=3, linewidth=180)

    problem = GridProblem(8)

    policy, values = policy_iteration(problem, 0.9, 0.001)

    problem.print_policy(policy)
    problem.print_values(values)

    policy, values = value_iteration(problem, 0.9, 0.001)

    problem.print_policy(policy)
    problem.print_values(values)

if __name__ == "__main__":
    main()

#!/usr/bin/python3
import numpy as np

from learning.model_free import Problem

from learning.model_free import sarsa
from learning.model_free import qlearning
from learning.model_free import mc_value_iteration
from learning.model_free import sarsa_lambda
from learning.model_free import q_lambda

from learning.model_building import dyna_q_learning
from learning.model_building import dyna_q_learning_last_visit
# from learning.model_building import dyna_q_learning_stochastic


class WindyGridWorld(Problem):

    def __init__(self):
        self.m = 7
        self.n = 10
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        # self.wind = np.zeros(10, dtype = int)

        self.init_actions()

        Problem.__init__(self, self.m * self.n, 4)

    def sample_initial_state(self):
        return self.coord_to_state(3, 0)

    def init_actions(self):
        self._actions = []
        for s in range(self.m * self.n):
            s_actions = []
            i, j = self.state_to_coord(s)

            if self.valid_coord(i + 1, j):
                s_actions.append(0)
            if self.valid_coord(i - 1, j):
                s_actions.append(1)
            if self.valid_coord(i, j + 1):
                s_actions.append(2)
            if self.valid_coord(i, j - 1):
                s_actions.append(3)

            self._actions.append(s_actions)

        self._action_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def actions(self, s):
        return self._actions[s]

    def state_reward(self, s, a):
        if a not in self._actions[s]:
            raise Exception('State {0} does not allow action {1}'.format(s, a))

        i, j = self.state_to_coord(s)
        di, dj = self._action_offsets[a]

        nexti, nextj = i + di, j + dj
        nexti = max(0, nexti - self.wind[j])

        nexts = self.coord_to_state(nexti, nextj)

        return (nexts, -1)

    def is_final(self, s):
        return s == self.coord_to_state(3, 7)

    def state_to_coord(self, s):
        return (s // self.n, s % self.n)

    def coord_to_state(self, i, j):
        return i * self.n + j

    def valid_coord(self, i, j):
        return i >= 0 and i < self.m \
            and j >= 0 and j < self.n

    def print_policy(self, pi):
        pi = pi.reshape((self.m, self.n))

        actions = ['V ', '^ ', '> ', '< ']
        for i in range(self.m):
            for j in range(self.n):
                if self.is_final(self.coord_to_state(i, j)):
                    print("* ", end='')
                else:
                    print(actions[pi[i, j]], end='')
            print('')

    def print_values(self, v):
        np.set_printoptions(precision=2)
        print(v.reshape((self.m, self.n)))


def main():
    problem = WindyGridWorld()

    pi, v = sarsa(problem, 1000, epsilon=0.1, alpha=0.1, gamma=1.0)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = qlearning(problem, 10000, epsilon=0.1, alpha=0.1, gamma=1.0)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = sarsa_lambda(problem, 1000, epsilon=0.1, alpha=0.1, gamma=1.0)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = q_lambda(problem, 1000, epsilon=0.1, alpha=0.1, gamma=1.0)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = mc_value_iteration(problem, 1000, 10000, 0.2)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = dyna_q_learning(problem, 30, 50, epsilon=0.1, alpha=0.1, gamma=0.9)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = dyna_q_learning_last_visit(
        problem, 30, 50, epsilon=0.1, alpha=0.1, gamma=0.9, kappa=0.00)

    problem.print_policy(pi)
    problem.print_values(v)

#     pi, v = dyna_q_learning_stochastic(problem, 30, 50, epsilon = 0.1,
#       alpha = 0.1, gamma = 0.9)
#
#     problem.print_policy(pi)
#     problem.print_values(v)


if __name__ == "__main__":
    main()

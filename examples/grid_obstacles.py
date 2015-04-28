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
from learning.model_building import dyna_q_learning_stochastic


class GridObstacles(Problem):

    def __init__(self):
        self.m = 6
        self.n = 9

        self.obstacles = np.zeros((self.m, self.n), dtype=np.int)
        self.obstacles[1:4, 2] = 1.0
        self.obstacles[4, 5] = 1.0
        self.obstacles[0:3, 7] = 1.0

        self.start = self.coord_to_state(2, 0)
        self.goal = self.coord_to_state(0, 8)

        self.init_actions()

        Problem.__init__(self, self.m * self.n, 4)

    def sample_initial_state(self):
        return self.start

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
        nexts = self.coord_to_state(nexti, nextj)

        if not self.is_final(s) and self.is_final(nexts):
            return (nexts, 1.0)
        else:
            return (nexts, 0.0)

    def is_final(self, s):
        return s == self.goal

    def state_to_coord(self, s):
        return (s / self.n, s % self.n)

    def coord_to_state(self, i, j):
        return i * self.n + j

    def valid_coord(self, i, j):
        return i >= 0 and i < self.m \
            and j >= 0 and j < self.n \
            and not self.obstacles[i, j]

    def print_policy(self, pi):
        pi = pi.reshape((self.m, self.n))

        actions = ['v', '^', '>', '<']

        for i in range(self.m):
            for j in range(self.n):
                if self.is_final(self.coord_to_state(i, j)):
                    print("*"),
                elif self.start == self.coord_to_state(i, j):
                    print("*"),
                elif self.obstacles[i, j]:
                    print("-"),
                else:
                    print(actions[pi[i, j]]),
            print('')

    def print_values(self, v):
        np.set_printoptions(precision=2)
        print(v.reshape((self.m, self.n)))


def main():
    problem = GridObstacles()
    pi, v = sarsa(problem, 1000, epsilon=0.1, alpha=0.1, gamma=1.0)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = qlearning(problem, 1000, epsilon=0.1, alpha=0.1, gamma=1.0)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = sarsa_lambda(problem, 1000, epsilon=0.1, alpha=0.1, gamma=1.0)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = q_lambda(problem, 1000, epsilon=0.1, alpha=0.1, gamma=1.0)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = mc_value_iteration(problem, 1000, 1000, 0.2)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = dyna_q_learning(problem, 30, 50, epsilon=0.1, alpha=0.1, gamma=0.9)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = dyna_q_learning_last_visit(
        problem, 30, 50, epsilon=0.1, alpha=0.1, gamma=0.9, kappa=0.00)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = dyna_q_learning_stochastic(
        problem, 30, 50, epsilon=0.1, alpha=0.1, gamma=0.9)

    problem.print_policy(pi)
    problem.print_values(v)

if __name__ == "__main__":
    main()

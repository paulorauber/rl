import numpy as np


class StatelessProblem:

    def __init__(self, n):
        self.n = n

    def reward(self, a):
        return 0

    def optimal_actions(self):
        return [0]


class Agent:

    def __init__(self, action_selection, update):
        self.action_selection = action_selection
        self.update = update

    def play(self, problem, iterations):
        Qt_prev = np.zeros(problem.n, float)
        k = np.zeros(problem.n, int)

        rewards = np.zeros(iterations, float)
        optimals = np.zeros(iterations, bool)

        actions = list(range(len(Qt_prev)))

        for i in range(iterations):
            action = self.action_selection(Qt_prev, actions)
            rewards[i] = problem.reward(action)

            k[action] += 1
            Qt_prev[action] = self.update(
                Qt_prev[action], rewards[i], k[action])

            if action in problem.optimal_actions:
                optimals[i] = True

        return rewards, optimals

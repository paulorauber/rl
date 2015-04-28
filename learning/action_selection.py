import numpy as np


def argmax_random(A):
    arg = np.argsort(A)[::-1]
    n_tied = sum(np.isclose(A, A[arg[0]]))
    return np.random.choice(arg[0:n_tied])


class EGreedySelection:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, q, actions):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(actions)
        else:
            return actions[argmax_random(q[actions])]


class SoftMaxSelection:

    def __init__(self, tau):
        self.tau = tau

    def __call__(self, q, actions):
        values = np.exp(q[actions] / self.tau)
        values = values / np.sum(values)

        return np.random.choice(actions, p=values)

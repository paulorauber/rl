import numpy as np


class Problem:

    def __init__(self):
        self.states = []

    def p(self, s, next_s, action):
        return 0

    def r(self, s, next_s, action):
        return 0

    def a(self, s):
        return []


def value_iteration(problem, gamma=0.9, theta=0.001):
    value = np.zeros(len(problem.states))

    p = problem.p
    r = problem.r

    while True:
        delta = 0
        for s in problem.states:
            actions = problem.a(s)
            v = value[s]
            value[s] = max([sum([p(s, next_s, a) * (r(s, next_s, a) + gamma * value[next_s])
                                 for next_s in problem.states]) for a in actions])
            delta = max(delta, abs(v - value[s]))
        if delta < theta:
            break

    policy = np.zeros(len(problem.states), dtype=int)
    for s in problem.states:
        actions = problem.a(s)
        policy[s] = actions[np.argmax(
            [sum([p(s, next_s, a) * (r(s, next_s, a) + gamma * value[next_s]) for next_s in problem.states]) for a in actions])]

    return policy, value


def eval_policy(problem, policy, gamma=0.9, theta=0.01):
    value = np.zeros(len(problem.states))

    p = problem.p
    r = problem.r

    while True:
        delta = 0
        for s in problem.states:
            v = value[s]
            value[s] = sum([p(s, next_s, policy[s]) * (r(s, next_s, policy[s]) + gamma * value[next_s]) for next_s in problem.states])

            delta = max(delta, abs(v - value[s]))

        if delta < theta:
            return value


def improve_policy(problem, policy, value, gamma=0.9):
    p = problem.p
    r = problem.r

    stable = True
    for s in problem.states:
        actions = problem.a(s)

        b = policy[s]
        policy[s] = actions[np.argmax(
            [sum([p(s, next_s, a) * (r(s, next_s, a) + gamma * value[next_s]) for next_s in problem.states]) for a in actions])]
        if b != policy[s]:
            stable = False

    return stable


def policy_iteration(problem, gamma=0.9, theta=0.01):
    policy = np.array([np.random.choice(problem.a(s)) for s in problem.states])

    stable = False
    while not stable:
        values = eval_policy(problem, policy, gamma, theta)
        stable = improve_policy(problem, policy, values, gamma)

    return policy, values

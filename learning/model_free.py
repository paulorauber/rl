import numpy as np
from learning.action_selection import EGreedySelection
from learning.action_selection import argmax_random


class Problem:

    def __init__(self, nstates, nactions):
        self.nstates = nstates
        self.nactions = nactions

    def sample_initial_state(self):
        return 0

    def actions(self, s):
        return [0]

    def state_reward(self, s, a):
        return (0, 0)

    def is_final(self, s):
        return True


def sarsa(problem, nepisodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    aselection = EGreedySelection(epsilon)

    nstates, nactions = problem.nstates, problem.nactions

    q = np.zeros((nstates, nactions))
    q.fill(float('-inf'))

    for s in range(nstates):
        actions = problem.actions(s)
        for a in actions:
            q[s, a] = 0

    for _ in range(nepisodes):
        s = problem.sample_initial_state()
        a = aselection(q[s], problem.actions(s))

        while not problem.is_final(s):
            next_s, r = problem.state_reward(s, a)
            next_a = aselection(q[next_s], problem.actions(next_s))

            q[s, a] = q[s, a] + alpha * \
                (r + gamma * q[next_s, next_a] - q[s, a])

            s = next_s
            a = next_a

    pi = q.argmax(axis=1)
    v = q.max(axis=1)

    return pi, v


def qlearning(problem, nepisodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    aselection = EGreedySelection(epsilon)

    nstates, nactions = problem.nstates, problem.nactions

    q = np.zeros((nstates, nactions))
    q.fill(float('-inf'))

    for s in range(nstates):
        actions = problem.actions(s)
        for a in actions:
            q[s, a] = 0

    for _ in range(nepisodes):
        s = problem.sample_initial_state()

        while not problem.is_final(s):
            a = aselection(q[s], problem.actions(s))

            next_s, r = problem.state_reward(s, a)
            q[s, a] = q[s, a] + alpha * (r + gamma * max(q[next_s]) - q[s, a])

            s = next_s

    pi = q.argmax(axis=1)
    v = q.max(axis=1)

    return pi, v


def mc_play(problem, pi, max_episode=100000):
    s = problem.sample_initial_state()

    rewards = []
    states_actions = []

    for _ in range(max_episode):
        a = np.random.choice(len(pi[s]), p=pi[s])

        states_actions.append((s, a))

        s, r = problem.state_reward(s, a)

        rewards.append(r)

        if problem.is_final(s):
            break

    return rewards, states_actions


def mc_value_iteration(problem, max_iter=10000, max_episode=100000,
                       epsilon=0.1):
    # Value estimate for state-action pair
    q = np.zeros((problem.nstates, problem.nactions))
    q.fill(float('-inf'))

    # Number of samples for state-action pair
    n = np.zeros((problem.nstates, problem.nactions))
    # Policy (e-soft)
    pi = np.zeros((problem.nstates, problem.nactions))

    for s in range(problem.nstates):
        actions = problem.actions(s)

        for a in actions:
            pi[s, a] = 1. / len(actions)
            q[s, a] = 0

    for _ in range(max_iter):
        rewards, states_actions = mc_play(problem, pi, max_episode)

        ret = sum(rewards)
        for j, (s, a) in enumerate(states_actions):
            n[s, a] += 1
            q[s, a] = q[s, a] + (1. / n[s, a]) * (ret - q[s, a])

            ret = ret - rewards[j]

        states = list({s for (s, a) in states_actions})
        a_stars = [argmax_random(q[s]) for s in states]

        for j, s in enumerate(states):
            action = a_stars[j]

            actions = problem.actions(s)
            for a in actions:
                if a == action:
                    pi[s, a] = 1. - epsilon + (epsilon / len(actions))
                else:
                    pi[s, a] = epsilon / len(actions)

    pi = q.argmax(axis=1)
    v = q.max(axis=1)

    return pi, v


def sarsa_lambda(problem, nepisodes, alpha=0.1, gamma=0.9, epsilon=0.1,
                 _lambda=0.9):
    # Accumulating traces
    aselection = EGreedySelection(epsilon)

    nstates, nactions = problem.nstates, problem.nactions

    q = np.zeros((nstates, nactions))
    q.fill(float('-inf'))
    et = np.zeros((nstates, nactions))

    for s in range(nstates):
        actions = problem.actions(s)
        for a in actions:
            q[s, a] = 0

    for _ in range(nepisodes):
        s = problem.sample_initial_state()
        a = aselection(q[s], problem.actions(s))

        et.fill(0)

        while not problem.is_final(s):
            next_s, r = problem.state_reward(s, a)
            next_a = aselection(q[next_s], problem.actions(next_s))

            et[s, a] += 1
            q = q + alpha * (r + gamma * q[next_s, next_a] - q[s, a]) * et
            et = et * gamma * _lambda

            s = next_s
            a = next_a

    pi = q.argmax(axis=1)
    v = q.max(axis=1)

    return pi, v


def q_lambda(problem, nepisodes, alpha=0.1, gamma=0.9, epsilon=0.1,
             _lambda=0.9):
    # Accumulating traces
    aselection = EGreedySelection(epsilon)

    nstates, nactions = problem.nstates, problem.nactions

    q = np.zeros((nstates, nactions))
    q.fill(float('-inf'))
    et = np.zeros((nstates, nactions))

    for s in range(nstates):
        actions = problem.actions(s)
        for a in actions:
            q[s, a] = 0

    for _ in range(nepisodes):
        s = problem.sample_initial_state()
        a = aselection(q[s], problem.actions(s))

        et.fill(0)
        while not problem.is_final(s):
            next_s, r = problem.state_reward(s, a)

            next_a = aselection(q[next_s], problem.actions(next_s))

            a_star_next = argmax_random(q[next_s])
            if np.allclose(q[next_s, next_a], q[next_s, a_star_next]):
                a_star_next = next_a

            et[s, a] += 1
            q = q + alpha * (r + gamma * q[next_s, a_star_next] - q[s, a]) * et
            if next_a == a_star_next:
                et = et * gamma * _lambda
            else:
                et.fill(0)

            s = next_s
            a = next_a

    pi = q.argmax(axis=1)
    v = q.max(axis=1)

    return pi, v

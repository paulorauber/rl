import numpy as np

from learning.action_selection import EGreedySelection
from learning.action_selection import argmax_random


def dyna_q_learning(problem, nepisodes, nplanning, epsilon, alpha, gamma):
    aselection = EGreedySelection(epsilon)

    q = np.zeros((problem.nstates, problem.nactions))
    q.fill(float('-inf'))

    model_next_r = np.zeros((problem.nstates, problem.nactions))
    model_next_s = np.zeros((problem.nstates, problem.nactions))

    visited = {}

    for s in range(problem.nstates):
        actions = problem.actions(s)
        for a in actions:
            q[s, a] = 0

    for _ in range(nepisodes):
        s = problem.sample_initial_state()

        while not problem.is_final(s):
            a = aselection(q[s], problem.actions(s))

            if s not in visited:
                visited[s] = set()
            visited[s].add(a)

            next_s, r = problem.state_reward(s, a)

            model_next_s[s, a] = next_s
            model_next_r[s, a] = r

            a_star_next = argmax_random(q[next_s])
            q[s, a] = q[s, a] + alpha * \
                (r + gamma * q[next_s, a_star_next] - q[s, a])

            real_next_s = next_s

            for _ in range(nplanning):
                s = np.random.choice(list(visited.keys()))
                a = np.random.choice(tuple(visited[s]))

                next_s, r = model_next_s[s, a], model_next_r[s, a]

                a_star_next = argmax_random(q[next_s])
                q[s, a] = q[s, a] + alpha * \
                    (r + gamma * q[next_s, a_star_next] - q[s, a])

            s = real_next_s

    pi = q.argmax(axis=1)
    v = q.max(axis=1)

    return pi, v


def dyna_q_learning_last_visit(problem, nepisodes, nplanning, epsilon, alpha,
                               gamma, kappa):
    aselection = EGreedySelection(epsilon)

    q = np.zeros((problem.nstates, problem.nactions))
    q.fill(float('-inf'))

    model_next_r = np.zeros((problem.nstates, problem.nactions))
    model_next_s = np.zeros((problem.nstates, problem.nactions))
    last_visit = np.zeros((problem.nstates, problem.nactions))

    visited = {}

    for s in range(problem.nstates):
        actions = problem.actions(s)
        for a in actions:
            q[s, a] = 0

    for i in range(nepisodes):
        s = problem.sample_initial_state()

        while not problem.is_final(s):
            a = aselection(q[s], problem.actions(s))

            if s not in visited:
                visited[s] = set()
            visited[s].add(a)
            last_visit[s, a] = i

            next_s, r = problem.state_reward(s, a)

            model_next_s[s, a] = next_s
            model_next_r[s, a] = r

            a_star_next = argmax_random(q[next_s])
            q[s, a] = q[s, a] + alpha * \
                (r + gamma * q[next_s, a_star_next] - q[s, a])

            real_next_s = next_s

            for _ in range(nplanning):
                s = np.random.choice(list(visited.keys()))
                a = np.random.choice(tuple(visited[s]))

                next_s, r = model_next_s[s, a], model_next_r[s, a]
                time_since_visit = i - last_visit[s, a]

                r += kappa * np.sqrt(time_since_visit)

                a_star_next = argmax_random(q[next_s])
                q[s, a] = q[s, a] + alpha * \
                    (r + gamma * q[next_s, a_star_next] - q[s, a])

            s = real_next_s

    pi = q.argmax(axis=1)
    v = q.max(axis=1)

    return pi, v


def dyna_q_learning_stochastic(problem, nepisodes, nplanning, epsilon, alpha,
                               gamma):
    aselection = EGreedySelection(epsilon)

    q = np.zeros((problem.nstates, problem.nactions))
    q.fill(float('-inf'))

    rssna = np.zeros((problem.nstates, problem.nstates, problem.nactions))
    nssna = np.zeros((problem.nstates, problem.nstates, problem.nactions))
    nsa = np.zeros((problem.nstates, problem.nactions))

    visited = {}

    for s in range(problem.nstates):
        actions = problem.actions(s)
        for a in actions:
            q[s, a] = 0

    for _ in range(nepisodes):
        s = problem.sample_initial_state()

        while not problem.is_final(s):
            a = aselection(q[s], problem.actions(s))

            if s not in visited:
                visited[s] = set()
            visited[s].add(a)

            next_s, r = problem.state_reward(s, a)

            # Model update
            nsa[s, a] += 1
            nssna[s, next_s, a] += 1
            rssna[s, next_s, a] = rssna[s, next_s, a] + \
                (r - rssna[s, next_s, a]) / nssna[s, next_s, a]

            a_star_next = argmax_random(q[next_s])
            q[s, a] = q[s, a] + alpha * \
                (r + gamma * q[next_s, a_star_next] - q[s, a])

            real_next_s = next_s

            for _ in range(nplanning):
                s = np.random.choice(list(visited.keys()))
                a = np.random.choice(tuple(visited[s]))

                q[s, a] = 0
                for sp in range(problem.nstates):
                    q[s, a] += (nssna[s, sp, a] / nsa[s, a]) * \
                        (rssna[s, sp, a] + gamma * max(q[sp]))

            s = real_next_s

    pi = q.argmax(axis=1)
    v = q.max(axis=1)

    return pi, v

#!/usr/bin/python3
import sys
import pylab

from learning.model_based import Problem
from learning.model_based import policy_iteration
from learning.model_based import value_iteration


class GamblerProblem(Problem):

    def __init__(self, goal=100, p_heads=0.4):
        self.goal = goal
        self.p_heads = p_heads
        self.states = range(goal + 1)

    def p(self, s, next_s, action):
        if action == 0 and s == next_s:
            return 1.

        if s == (next_s - action):
            return self.p_heads
        if s == (next_s + action):
            return 1 - self.p_heads

        return 0

    def r(self, s, next_s, action):
        if next_s == self.goal and s != next_s:
            return 1

        return 0

    def a(self, s):
        return range(0, min(s, self.goal - s) + 1)

    def print_policy(self, policy):
        pylab.scatter(range(len(policy)), policy)
        pylab.xlabel('credit')
        pylab.ylabel('bet')
        pylab.show()

    def print_values(self, values):
        pylab.scatter(range(len(values)), values)
        pylab.xlabel('credit')
        pylab.ylabel('expected return')
        pylab.show()


def main():
    p_heads = 0.4
    if len(sys.argv) > 1:
        p_heads = float(sys.argv[1])

    problem = GamblerProblem(100, p_heads)

    policy, values = value_iteration(problem, 1.0, 0.001)

    problem.print_policy(policy)
    problem.print_values(values)

    policy, values = policy_iteration(problem, 0.9, 0.001)

    problem.print_policy(policy)
    problem.print_values(values)

if __name__ == "__main__":
    main()

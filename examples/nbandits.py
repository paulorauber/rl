#!/usr/bin/python3
import numpy as np
import pylab

from learning.stateless import StatelessProblem
from learning.stateless import Agent
from learning.update_rules import SampleAverageUpdate
from learning.update_rules import ConstantStepUpdate
from learning.action_selection import EGreedySelection
from learning.action_selection import SoftMaxSelection


class NBandit(StatelessProblem):

    def __init__(self, n):
        super(NBandit, self).__init__(n)

        self.Qstar = np.random.normal(size=n)

        self.optimal_value = max(self.Qstar)
        self.optimal_actions = [
            i for i in range(self.n) if self.Qstar[i] == self.optimal_value]

    def reward(self, a):
        return np.random.normal(self.Qstar[a], 1.0, size=1)


class RandomNBandit(StatelessProblem):

    def __init__(self, n, std=0.1):
        super(RandomNBandit, self).__init__(n)

        self.Qstar = np.ones(n)
        self.std = std

    def reward(self, a):
        self.Qstar[a] = self.Qstar[a] + np.random.normal(0.0, self.std)

        self.optimal_value = max(self.Qstar)
        self.optimal_actions = [
            i for i in range(self.n) if self.Qstar[i] == self.optimal_value]
        return np.random.normal(self.Qstar[a], 1.0, size=1)


def experiment_1():
    niters = 1000
    nbandits = 2000
    narms = 10

    bandits = [NBandit(narms) for _ in range(nbandits)]

    egreedy = Agent(EGreedySelection(0.1), SampleAverageUpdate())
    egreedy_avg_rewards = np.zeros(niters, float)
    egreedy_avg_optimals = np.zeros(niters, float)

    softmax = Agent(SoftMaxSelection(0.2), SampleAverageUpdate())
    softmax_avg_rewards = np.zeros(niters, float)
    softmax_avg_optimals = np.zeros(niters, float)

    for nbandit in bandits:
        rewards, optimals = egreedy.play(nbandit, niters)
        egreedy_avg_rewards += rewards
        egreedy_avg_optimals += optimals

        rewards, optimals = softmax.play(nbandit, niters)
        softmax_avg_rewards += rewards
        softmax_avg_optimals += optimals

    egreedy_avg_rewards /= float(nbandits)
    egreedy_avg_optimals /= float(nbandits)

    softmax_avg_rewards /= float(nbandits)
    softmax_avg_optimals /= float(nbandits)

    pylab.title('Average reward per iteration')
    pylab.plot(range(niters), egreedy_avg_rewards, label='e-Greedy')
    pylab.plot(range(niters), softmax_avg_rewards, label='SoftMax')
    pylab.legend(loc='best')

    pylab.show()

    pylab.title('Average optimal actions per iteration')
    pylab.plot(range(niters), egreedy_avg_optimals, label='e-Greedy')
    pylab.plot(range(niters), softmax_avg_optimals, label='SoftMax')
    pylab.legend(loc='best')

    pylab.show()


def experiment_2():
    niters = 1000
    nbandits = 2000
    narms = 10

    egreedy_const = Agent(EGreedySelection(0.1), ConstantStepUpdate(step=0.1))
    egreedy_const_avg_rewards = np.zeros(niters, float)
    egreedy_const_avg_optimals = np.zeros(niters, float)

    egreedy_avg = Agent(EGreedySelection(0.1), SampleAverageUpdate())
    egreedy_avg_avg_rewards = np.zeros(niters, float)
    egreedy_avg_avg_optimals = np.zeros(niters, float)

    for _ in range(nbandits):
        nbandit = RandomNBandit(narms, std=0.05)
        rewards, optimals = egreedy_const.play(nbandit, niters)
        egreedy_const_avg_rewards += rewards
        egreedy_const_avg_optimals += optimals

        nbandit = RandomNBandit(narms, std=0.05)
        rewards, optimals = egreedy_avg.play(nbandit, niters)
        egreedy_avg_avg_rewards += rewards
        egreedy_avg_avg_optimals += optimals

    egreedy_const_avg_rewards /= float(nbandits)
    egreedy_const_avg_optimals /= float(nbandits)

    egreedy_avg_avg_rewards /= float(nbandits)
    egreedy_avg_avg_optimals /= float(nbandits)

    pylab.title('Average reward per iteration')
    pylab.plot(range(niters), egreedy_const_avg_rewards,
               label='e-Greedy constant update step')
    pylab.plot(range(niters), egreedy_avg_avg_rewards,
               label='e-Greedy sample average')
    pylab.legend(loc='best')

    pylab.show()

    pylab.title('Average optimal actions per iteration')
    pylab.plot(range(niters), egreedy_const_avg_optimals,
               label='e-Greedy constant update step')
    pylab.plot(range(niters), egreedy_avg_avg_optimals,
               label='e-Greedy sample average')
    pylab.legend(loc='best')

    pylab.show()


def main():
    experiment_1()
    experiment_2()

if __name__ == "__main__":
    main()

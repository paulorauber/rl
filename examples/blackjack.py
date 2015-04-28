import numpy as np
from itertools import product

from learning.model_free import Problem
from learning.model_free import sarsa
from learning.model_free import qlearning
from learning.model_free import mc_value_iteration
from learning.model_free import sarsa_lambda
from learning.model_free import q_lambda

# from learning.model_building import dyna_q_learning
# from learning.model_building import dyna_q_learning_last_visit
# from learning.model_building import dyna_q_learning_stochastic


class BlackJack(Problem):

    def __init__(self):
        # Sum of player's cards, dealer's showing card, usable ace
        self.states = [(-1, -1, -1)]
        self.states += [(i, j, k)
                        for (i, j, k) in product(range(12, 22), range(1, 11), [0, 1])]
        self.a = ['hit', 'stick']

        self.states_map = {s: i for i, s in enumerate(self.states)}

        Problem.__init__(self, len(self.states), len(self.a))

    def get_card(self):
        return min(10, np.random.randint(1, 14))

    def sample_initial_state(self):
        my_card = np.random.randint(12, 22)
        dealer_showing = self.get_card()
        usable_ace = np.random.randint(0, 2)

        return self.states_map[(my_card, dealer_showing, usable_ace)]

    def hand_value(self, sum_cards, usable_ace):
        if usable_ace and sum_cards > 21:
            return sum_cards - 10
        return sum_cards

    def actions(self, s):
        (my_sum, _, usable_ace) = self.states[s]

        if self.hand_value(my_sum, usable_ace) >= 21:
            return [1]
        else:
            return [0, 1]

    def is_final(self, s):
        return s == 0

    # Computes the next state and reward pair and whether the state is final
    def state_reward(self, s, a):
        (my_sum, dealer_card, usable_ace) = self.states[s]
        next_s = self.states_map[(my_sum, dealer_card, usable_ace)]

        if a == 1:  # Stick
            if self.hand_value(my_sum, usable_ace) > 21:
                return 0, -1

            dealer_sum = dealer_card
            dealer_usable_ace = 0
            if dealer_card == 1:
                dealer_sum += 10
                dealer_usable_ace = 1

            while self.hand_value(dealer_sum, dealer_usable_ace) < self.hand_value(my_sum, usable_ace):
                card = self.get_card()
                dealer_sum += card
                if card == 1:
                    dealer_sum += 10

                if card == 1 or dealer_usable_ace:
                    if dealer_sum <= 21:
                        dealer_usable_ace = 1
                    else:
                        dealer_sum -= 10
                        dealer_usable_ace = 0

                if self.hand_value(dealer_sum, dealer_usable_ace) == self.hand_value(my_sum, usable_ace) == 17:
                    return 0, 0

            if self.hand_value(dealer_sum, dealer_usable_ace) > 21:
                return 0, 1

            if self.hand_value(dealer_sum, dealer_usable_ace) == self.hand_value(my_sum, usable_ace):
                return 0, 0

            if self.hand_value(dealer_sum, dealer_usable_ace) < self.hand_value(my_sum, usable_ace):
                return 0, 1

            # if dealer_sum > my_sum:
            return 0, -1
        else:  # Hit
            card = self.get_card()
            my_sum += card
            if card == 1:
                my_sum += 10

            if card == 1 or usable_ace:
                if my_sum <= 21:
                    usable_ace = 1
                else:
                    my_sum -= 10
                    usable_ace = 0

            if self.hand_value(my_sum, usable_ace) > 21:
                return 0, -1

            # Only nonterminal case
            next_s = self.states_map[(my_sum, dealer_card, usable_ace)]
            return next_s, 0

        raise Exception('Unexpected state/action pair')

    def print_policy(self, policy):
        print('Usable ace:')
        for i, state in enumerate(self.states):
            if state[2]:
                print('Hand value: {0}, Dealer Showing: {1}, Action: {2}'.format(
                    self.hand_value(state[0], 1), state[1], self.a[policy[i]]))

        print('No usable ace:')
        for i, state in enumerate(self.states):
            if not state[2]:
                print('Hand value: {0}, Dealer Showing: {1}, Action: {2}'.format(
                    self.hand_value(state[0], 0), state[1], self.a[policy[i]]))

    def print_values(self, values):
        for i in range(len(values)):
            print('State {0}. Value: {1}'.format(self.states[i], values[i]))


def main():
    problem = BlackJack()

    pi, v = sarsa(problem, 10000, epsilon=0.1, alpha=0.1, gamma=1.0)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = qlearning(problem, 10000, epsilon=0.1, alpha=0.1, gamma=1.0)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = sarsa_lambda(problem, 10000, epsilon=0.1, alpha=0.1, gamma=1.0)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = q_lambda(problem, 10000, epsilon=0.1, alpha=0.1, gamma=1.0)

    problem.print_policy(pi)
    problem.print_values(v)

    pi, v = mc_value_iteration(problem, 10000, 10000, 0.2)

    problem.print_policy(pi)
    problem.print_values(v)

    # pi, v = dyna_q_learning(problem, 30, 50, epsilon = 0.1, alpha = 0.1, gamma = 0.9)
    #
    # problem.print_policy(pi)
    # problem.print_values(v)
    #
    # pi, v = dyna_q_learning_last_visit(problem, 30, 50, epsilon = 0.1, alpha = 0.1, gamma = 0.9, kappa = 0.00)

    # problem.print_policy(pi)
    # problem.print_values(v)
    #
    # pi, v = dyna_q_learning_stochastic(problem, 30, 50, epsilon = 0.1, alpha = 0.1, gamma = 0.9)
    #
    # problem.print_policy(pi)
    # problem.print_values(v)

if __name__ == "__main__":
    main()

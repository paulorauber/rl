import numpy as np

import argparse

"""This example is incomplete"""


class Checkers:

    def __init__(self):
        self.nfeats = 64

    # Black pieces begin
    def sample_initial_state(self):
        p1 = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        p2 = 1 - p1

        return np.array([p1, p2, p1, np.zeros(8), np.zeros(8), -p2, -p1, -p2])

    # wrt black
    def next_states(self, s, continue_from=None):
        next_states = []

        man_moves = [(1, 1), (1, -1)]
        king_moves = [(1, -1), (1, 1), (-1, -1), (-1, 1)]

        # Makes a move using a piece that already jumped
        if continue_from:
            i, j = continue_from[0], continue_from[1]
            v = s[i, j]

            moves = []

            if v == 1:
                moves = man_moves
            if v == 2:
                moves = king_moves

            found_valid = False
            for di, dj in moves:
                nexti = i + di
                nextj = j + dj

                if self.valid_coord(nexti, nextj):
                    if s[nexti, nextj] < 0:
                        if self.valid_coord(nexti + di, nextj + dj):
                            if s[nexti + di, nextj + dj] == 0:
                                s_next = np.copy(s)

                                s_next[i, j] = 0
                                s_next[nexti, nextj] = 0
                                s_next[nexti + di, nextj + dj] = v
                                if nexti + di == 7:
                                    s_next[nexti + di, nextj + dj] = 2

                                continue_from = (nexti + di, nextj + dj)
                                next_states.extend(
                                    self.next_states(s_next, continue_from))

                                found_valid = True

            if not found_valid:
                next_states.append(s)
        else:
            # Makes a move using a piece that hasn't jumped
            for (i, j), v in np.ndenumerate(s):
                moves = []

                if v == 1:
                    moves = man_moves
                if v == 2:
                    moves = king_moves

                for di, dj in moves:
                    nexti = i + di
                    nextj = j + dj
                    if self.valid_coord(nexti, nextj):
                        if s[nexti, nextj] < 0:
                            if self.valid_coord(nexti + di, nextj + dj):
                                if s[nexti + di, nextj + dj] == 0:
                                    s_next = np.copy(s)

                                    s_next[i, j] = 0
                                    s_next[nexti, nextj] = 0
                                    s_next[nexti + di, nextj + dj] = v
                                    if nexti + di == 7:
                                        s_next[nexti + di, nextj + dj] = 2

                                    continue_from = (nexti + di, nextj + dj)
                                    next_states.extend(
                                        self.next_states(s_next, continue_from))

        if not continue_from:
            # Makes a simple move
            for (i, j), v in np.ndenumerate(s):
                moves = []

                if v == 1:
                    moves = man_moves
                if v == 2:
                    moves = king_moves

                for di, dj in moves:
                    nexti = i + di
                    nextj = j + dj
                    if self.valid_coord(nexti, nextj):
                        if s[nexti, nextj] == 0:
                            s_next = np.copy(s)
                            s_next[i, j] = 0
                            s_next[nexti, nextj] = v
                            if nexti == 7:
                                s_next[nexti, nextj] = 2

                            next_states.append(s_next)

            # There are no moves whatsoever
            if not next_states:
                return [s]

        return next_states

    def valid_coord(self, i, j):
        return 0 <= i < 8 and 0 <= j < 8

    def flip(self, s):
        return np.flipud(-s)

    def print_state(self, s):
        for (_, j), v in np.ndenumerate(s):
            if v == 1:
                print('b'),
            elif v == 2:
                print('B'),
            elif v == -1:
                print('w'),
            elif v == -2:
                print('W'),
            else:
                print('-'),

            if j == 7:
                print('')
        print('')


def run_training():
    pass


def run_demo():
    pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--training", action="store_true", help="train the agent")
    parser.add_argument("-d", "--demo", action="store_true",
                        help="step through a game between the agent and itself")

    args = parser.parse_args()

    if args.training:
        run_training()
    elif args.demo:
        run_demo()


if __name__ == "__main__":
    main()

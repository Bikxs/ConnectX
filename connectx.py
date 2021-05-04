from random import random
from typing import List

import numpy as np


def is_subsequence(a: List, b: List):
    if not a:
        return True
    if not b:
        return False
    diff_len = len(b) - len(a)
    if diff_len < 0:
        return False
    for x in range(diff_len + 1):
        result = [_a == b[index + x] for index, _a in enumerate(a)]
        if all(result):
            return True
    return False


def alphabeta(game, time_left, depth, alpha=float("-inf"), beta=float("inf"), my_turn=True):
    """Implementation of the minimax algorithm.
    Args:
        player (CustomPlayer): This is the instantiation of CustomPlayer()
            that represents your agent. It is used to call anything you
            need from the CustomPlayer class (the utility() method, for example,
            or any class variables that belong to CustomPlayer()).
        game (Board): A board and game state.
        time_left (function): Used to determine time left before timeout
        depth: Used to track how deep you are in the search tree
        my_turn (bool): True if you are computing scores during your turn.

    Returns:
        (tuple, int): best_move, val
    """
    moves = game.moves_available()  # game.get_player_moves(active_player)
    if not moves:
        return None, 0
        # raise Exception("No possible move left")
    best_move = None
    new_depth = depth - 1
    if my_turn:
        # my turn
        value = -float('inf')
        for column_index in moves:
            won, new_game = game.forecast_move(column_index)
            if won:
                return column_index, new_game.score(game.player)
            else:
                if new_depth == 0 or time_left() < 100:
                    score = new_game.score(game.player)
                else:
                    _, score = alphabeta(new_game, time_left, new_depth, my_turn=False)
                if score > value:
                    best_move = column_index
                    value = score
                elif score == value and random() < 1.0/len(moves):
                    best_move = column_index
                    value = score
                # beta pruning
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        return best_move, value
    else:
        # opponent's turn
        value = float('inf')
        for column_index in moves:
            won, new_game = game.forecast_move(column_index)
            if won:
                return column_index, -new_game.score(game.player)
            else:
                if new_depth == 0 or time_left() < 100:
                    score = -new_game.score(game.player)
                else:
                    _, score = alphabeta(new_game, time_left, new_depth, my_turn=True)
                if score < value:
                    best_move = column_index
                    value = score
                elif score == value and random() < 1.0/len(moves):
                    best_move = column_index
                    value = score
                #alpha prunning
                beta = min(beta, value)
                if beta <= alpha:
                    break
        return best_move, value
    return best_move, val


class ConnectX:
    def __init__(self, inarow, board, player):
        self.inarow = inarow
        self.player = player
        self.board = board

    def _check_move(self, col_index):
        column = self.board.T[col_index].tolist()
        if column[0] != 0:
            # raise Exception(f'Cannot make move is selected column {col_index}')
            return False, self
        for row_index, value in enumerate(column):
            if value == 0 and row_index == (len(column) - 1):
                board = self.board.copy()
                board[row_index, col_index] = self.player
                return True, ConnectX(self.inarow, board, player=1 if self.player == 2 else 2)
            elif value == 0 and column[row_index + 1] != 0:
                board = self.board.copy()
                board[row_index, col_index] = self.player
                return True, ConnectX(self.inarow, board, player=1 if self.player == 2 else 2)

        return False, self

    def moves_available(self):

        columns = self.board.T.tolist()
        moves = [(index, self._check_move(index)) for index, column in enumerate(columns)]
        return [column_index for column_index, (possible, game) in moves if possible]

    def forecast_move(self, col_index):
        possible, new_game = self._check_move(col_index=col_index)
        if not possible:
            raise Exception("Impossible move requested")

        return new_game.is_won(self.player), new_game

    def is_won(self, player):
        win_sequence = [player for _ in range(self.inarow)]
        rows = self.board.tolist()
        for row in rows:
            if is_subsequence(win_sequence, row):
                return True
        columns = self.board.T.tolist()
        for column in columns:
            if is_subsequence(win_sequence, column):
                return True
        diags = [self.board[::-1, :].diagonal(i) for i in range(-self.board.shape[0] + 1, self.board.shape[1])]
        diags.extend(self.board.diagonal(i) for i in range(self.board.shape[1] - 1, -self.board.shape[0], -1))
        diagonals = [n.tolist() for n in diags if len(n) >= self.inarow]
        for diagonal in diagonals:
            if diagonal:
                if is_subsequence(win_sequence, diagonal):
                    return True
        return False

    def score(self, player):
        opponent = 2 if player == 1 else 1
        if self.is_won(player):
            return 100
        elif self.is_won(opponent):
            return -100
        return 0

    @classmethod
    def fromEnv(cls, observation, configuration):
        num_columns = configuration['columns']
        num_rows = configuration['rows']
        board = np.array(observation['board'])
        inarow = configuration['inarow']
        board = np.array(board).reshape((num_rows, num_columns))
        current_player = observation['mark']
        return ConnectX(inarow=inarow, player=current_player, board=board)


if __name__ == '__main__':
    print("Invisible threads are the strongest ties.\n\n")
    print("Nothing to run - execute from jupyter notebook")

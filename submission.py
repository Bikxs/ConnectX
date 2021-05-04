import random
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
        mark (CustomPlayer): This is the instantiation of CustomPlayer()
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
    best_move = random.choice(moves)
    new_depth = depth - 1
    if my_turn:
        # my turn
        value = -float('inf')
        for column_index in moves:
            won, new_game = game.forecast_move(column_index)
            if won:
                return column_index, new_game.score(game.mark)
            else:
                if new_depth == 0 or time_left() < 100:
                    score = new_game.score(game.mark)
                else:
                    _, score = alphabeta(new_game, time_left, new_depth, my_turn=False)
                if score > value:
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
                return column_index, -new_game.score(game.mark)
            else:
                if new_depth == 0 or time_left() < 100:
                    score = -new_game.score(game.mark)
                else:
                    _, score = alphabeta(new_game, time_left, new_depth, my_turn=True)
                if score < value:
                    best_move = column_index
                    value = score
                # alpha prunning
                beta = min(beta, value)
                if beta <= alpha:
                    break
        return best_move, value
    return best_move, val


# Calculates score if agent drops piece in selected column
def score_move(grid, col, mark, config):
    next_grid = drop_piece(grid, col, mark, config)
    score = get_heuristic(next_grid, col, mark, config)
    return score


# Helper function for score_move: gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, mark, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid


# Helper function for get_heuristic: checks if window satisfies heuristic conditions
def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs)


# Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
def count_windows(grid, num_discs, piece, config):
    num_windows = 0
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[row, col:col + config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # vertical
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            window = list(grid[row:row + config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # positive diagonal
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # negative diagonal
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows


def get_heuristic(grid, mark, config):
    A = 1000000
    B = 100
    C = 1
    D = -100
    E = -1000
    num_twos = count_windows(grid, 2, mark, config)
    num_threes = count_windows(grid, 3, mark, config)
    num_fours = count_windows(grid, 4, mark, config)
    num_twos_opp = count_windows(grid, 2, mark % 2 + 1, config)
    num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)
    score = A * num_fours + B * num_threes + C * num_twos + D * num_twos_opp + E * num_threes_opp
    return num_fours > 0, score


class ConnectX:
    def __init__(self, board, mark, config):
        self.board = board
        self.config = config
        self.mark = mark

    def moves_available(self):
        top_row = self.board[0, :]
        return [column_index for column_index, value in enumerate(top_row) if value == 0]

    def forecast_move(self, col):
        if not col in self.moves_available():
            raise Exception("Impossible move requested")
        board = drop_piece(self.board, col=col, mark=self.mark, config=self.config)

        won, _ = get_heuristic(board, self.mark, config=self.config)
        mark = 1 if self.mark == 2 else 2
        return won, ConnectX(board, config=self.config, mark=mark)

    def score(self, mark):
        won, score = get_heuristic(self.board, mark=mark, config=self.config)
        return score

    @classmethod
    def fromEnv(cls, observation, configuration):
        num_columns = configuration['columns']
        num_rows = configuration['rows']
        mark = observation['mark']
        board = np.array(observation['board'])
        board = np.array(board).reshape((num_rows, num_columns))
        return ConnectX(board=board, config=configuration, mark=mark)

    def is_won(self, mark):
        won, score = get_heuristic(self.board, mark=mark, config=self.config)
        return won


def act(observation, configuration):
    # print(configuration)
    step = observation.step

    actTimeout = configuration.actTimeout

    from datetime import datetime, timedelta

    t0 = datetime.now() + timedelta(seconds=actTimeout)

    def time_left():
        return 1000 * (t0 - datetime.now()).total_seconds()

    game = ConnectX.fromEnv(observation=observation,
                            configuration=configuration)
    max_depth = 12
    move, value = alphabeta(game, time_left=time_left, depth=max_depth)
    # print("-------------------------------------------------------------------------")
    # print(game.board)
    # print()
    # print(f'{step}) Move:{move} Value:{value} by {game.mark}')

    return move

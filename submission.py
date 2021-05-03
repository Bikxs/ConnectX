from pprint import pprint

from connectx import ConnectX


def act(observation, configuration):
    def time_left():
        return 100

    board = observation.board
    columns = configuration.columns

    game = ConnectX.fromEnv(observation=observation,
                            configuration=configuration)
    print()
    pprint(game.board)
    moves = game.moves_available()
    move, value = _alphabeta(game, time_left=time_left, depth=4)
    print('moves:', moves)
    print(f'best move:{move} value {value}')
    random_move = [c for c in range(columns) if board[c] == 0][0]
    print('random_move:', random_move)
    print()
    # alpha-beta search to get best move
    return random_move


def _alphabeta(game, time_left, depth, alpha=float("-inf"), beta=float("inf"), my_turn=True):
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
        breakpoint()
    best_move = None
    new_depth = depth - 1
    if my_turn:
        # my turn
        value = -float('inf')
        for column_index, winning in enumerate(moves):
            if winning is None:
                raise Exception('Move not possible')
                continue
            new_board, is_over, winner = game.forecast_move(column_index)
            if is_over:
                return column_index, 100
            else:
                if new_depth == 0 or time_left() < 100:
                    score = new_board.score()
                else:
                    _, score = _alphabeta(new_board, time_left, new_depth, False)
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
        for column_index, winning in enumerate(moves):
            new_board, is_over, winner = game.forecast_move(column_index)
            if is_over:
                return column_index, -100
            else:
                if new_depth == 0 or time_left() < 100:
                    score = -new_board.score()
                else:
                    _, score = _alphabeta(new_board, time_left, new_depth, True)
                if score < value:
                    best_move = column_index
                    value = score
                    # alpha pruning
                beta = min(beta, value)
                if beta <= alpha:
                    break
        return best_move, value
    return best_move, val

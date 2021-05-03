import numpy as np


def _check_won(inarow: int, board: np.ndarray, player: int):
    rows = board.T.tolist()
    for row in rows:
        if row.count(player) >= inarow:
            return True
    columns = board.T.tolist()
    for column in columns:
        if column.count(player) >= inarow:
            return True
    diags = [board[::-1, :].diagonal(i) for i in range(-board.shape[0] + 1, board.shape[1])]
    diags.extend(board.diagonal(i) for i in range(board.shape[1] - 1, -board.shape[0], -1))
    diagonals = [n.tolist() for n in diags if len(n) >= inarow]
    for diagonal in diagonals:
        if diagonal:
            if diagonal.count(diagonal) >= inarow:
                return True
    return False


class ConnectX:
    def __init__(self, inarow, current_player, board):
        self.inarow = inarow
        self.current_player = current_player
        self.board = board

    def moves_available(self):
        def check_move(col_index, column):
            if column[0] != 0:
                return None
            for row_index, value in enumerate(column):
                do_row = (value == 0 and column[row_index + 1] != 0) or (
                        column[row_index + 1] == 0 and row_index == len(column) - 2)
                if do_row:
                    board = self.board.copy()
                    board[(row_index - 1), col_index] = self.current_player
                    # print(self.board)
                    # print(board)
                    # print()
                    return _check_won(self.inarow, board, self.current_player)
            return False

        columns = self.board.T.tolist()
        return [check_move(index, column) for index, column in enumerate(columns)]

    def forecast_move(self, col_index):
        def make_move(col_index):
            column = self.board.T[col_index].tolist()
            if column[0] != 0:
                raise Exception(f'Cannot make move is selected column {col_index}')
            for row_index, value in enumerate(column):
                do_row = (value == 0 and column[row_index + 1] != 0) or (
                        column[row_index + 1] == 0 and row_index == len(column) - 2)
                if do_row:
                    board = self.board.copy()
                    board[(row_index - 1), col_index] = self.current_player

                    return _check_won(self.inarow, board, self.current_player), board
            return False

        # new board is returned with players' turn updated (toggled)

        won, board = make_move(col_index=col_index)

        current_player = 1 if self.current_player == 2 else 1
        new_game = ConnectX(self.inarow,current_player=current_player,board=board)

        return new_game, won, self.current_player if won else None

    def score(self):
        return 0

    @classmethod
    def fromEnv(cls, observation, configuration):
        num_columns = configuration['columns']
        num_rows = configuration['rows']
        board = observation['board']
        inarow = configuration['inarow']
        board = np.array(board).reshape((num_rows, num_columns))
        current_player = observation['mark']
        return ConnectX(inarow, current_player, board)


if __name__ == '__main__':
    print("Invisible threads are the strongest ties.\n\n")
    print("Nothing to run - execute from jupyter notebook")

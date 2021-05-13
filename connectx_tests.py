import unittest

import numpy as np

from strategy_minimax import is_subsequence, Game, alphabeta


class Config:
    def __init__(self,
                 episodeSteps=1000,
                 actTimeout=2,
                 runTimeout=1200,
                 columns=7,
                 rows=6,
                 inarow=4,
                 agentTimeout=60,
                 timeout=2
                 ):
        self.episodeSteps = episodeSteps
        self.actTimeout = actTimeout
        self.runTimeout = runTimeout
        self.columns = columns
        self.rows = rows
        self.inarow = inarow
        self.agentTimeout = agentTimeout
        self.timeout = timeout


config = Config()
positions = [
    ("Empty",
     np.array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]]), False, False, 0, 0),

    ("P1 won horizontally",
     np.array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [2, 2, 2, 0, 0, 0, 0],
               [1, 1, 1, 1, 2, 2, 0]]), True, False, 100, -100),

    ("P2 won horizontally",
     np.array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 0, 0, 0, 0],
               [0, 1, 1, 0, 0, 0, 0],
               [2, 2, 2, 2, 0, 0, 0]]), False, True, -100, 100),

    ("P1 won vertically",
     np.array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0, 0],
               [0, 0, 1, 2, 0, 0, 0],
               [0, 2, 1, 2, 2, 0, 0],
               [1, 1, 2, 1, 0, 0, 0]]), True, False, 100, -100),

    ("P2 won vertically",
     np.array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 2, 0, 0, 0],
               [1, 2, 0, 2, 0, 0, 0],
               [1, 1, 2, 2, 0, 0, 0],
               [2, 1, 1, 2, 1, 1, 0]]), False, True, -100, 100),

    ("P2 won horizontally 2",
     np.array([[1, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 0],
               [2, 0, 0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0, 0, 1],
               [2, 2, 0, 2, 2, 2, 2],
               [1, 1, 1, 2, 1, 1, 2]])
     , False, True, -100, 100),

    ("P1 won diagonally",
     np.array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0],
               [0, 0, 1, 2, 0, 0, 0],
               [0, 1, 1, 2, 0, 0, 0],
               [1, 1, 2, 1, 0, 0, 0]]), True, False, 100, -100),

    ("P2 won diagonally",
     np.array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [2, 0, 0, 0, 0, 0, 0],
               [1, 2, 0, 0, 0, 0, 0],
               [1, 1, 2, 0, 0, 0, 0],
               [2, 1, 1, 2, 1, 1, 0]]), False, True, -100, 100),

    ("P2 won diagonally 2",
     np.array([[2, 2, 1, 1, 2, 0, 0],
               [1, 1, 2, 2, 1, 2, 0],
               [2, 2, 1, 1, 2, 1, 0],
               [1, 1, 2, 2, 1, 1, 0],
               [2, 2, 2, 1, 1, 2, 0],
               [1, 1, 2, 2, 1, 1, 2]]), False, True, -100, 100),

    ("P2 about to win",
     np.array([[2, 2, 1, 1, 2, 0, 0],
               [1, 1, 2, 2, 1, 0, 0],
               [2, 2, 1, 1, 2, 0, 0],
               [1, 1, 2, 2, 1, 1, 0],
               [2, 2, 2, 1, 1, 2, 0],
               [1, 1, 2, 2, 1, 1, 2]]), False, False, 0, 0),

]


class ConnectXTestCases(unittest.TestCase):
    def test_checkBoardScoreWon(self):
        for index, (board_description, board, won_P1, won_P2, score_P1, score_P2) in enumerate(positions):
            if index != 1:
                continue
            game1 = Game(board=board, mark=1, config=config)
            game2 = Game(board=board, mark=2, config=config)
            self.assertEqual(game1.is_won(1), won_P1,
                             msg=f'{index}) Did P1 win on board: {board_description}')
            self.assertEqual(game2.is_won(2), won_P2,
                             msg=f'{index}) Did P2 win on board: {board_description}')

    def test_subsequence1(self):
        a = [1, 1, 1, 1]
        b = [2, 2, 0, 1, 1, 1, 1, 2]
        self.assertTrue(is_subsequence(a, b), msg="Is subsequence")

    def test_subsequence2(self):
        a = [1, 1, 1, 1]
        b = [2, 2, 0, 1, 1, 0, 1, 1, 2]
        self.assertFalse(is_subsequence(a, b), msg="Is not subsequence")

    def test_subsequence3(self):
        a = [2, 2, 2, 2]
        b = [0, 0, 2, 2, 2, 2]
        self.assertTrue(is_subsequence(a, b), msg="Is subsequence")

    def test_alpha_beta_1(self):
        board = np.array([[2, 2, 1, 1, 2, 0, 0],
                          [1, 1, 2, 2, 1, 0, 0],
                          [2, 2, 1, 1, 2, 0, 0],
                          [1, 1, 2, 2, 1, 1, 0],
                          [2, 2, 2, 1, 1, 2, 0],
                          [1, 1, 2, 2, 1, 1, 2]])
        game = Game(board=board, mark=1, config=config)

        move, score = alphabeta(game, time_left=lambda: 100, depth=3)
        print(move)
        print(score)
        self.assertEqual(6, move, msg="P1 Move should be 6")
        # self.assertEqual(0, score, msg="P1 Score should be 0")

    def test_alpha_beta_2(self):
        board = np.array([[2, 2, 1, 1, 2, 0, 0],
                          [1, 1, 2, 2, 1, 0, 0],
                          [2, 2, 1, 1, 2, 1, 0],
                          [1, 1, 2, 2, 1, 1, 0],
                          [2, 2, 2, 1, 1, 2, 0],
                          [1, 1, 2, 2, 1, 1, 2]])
        game = Game(board=board, mark=2, config=config)

        move, score = alphabeta(game, time_left=lambda: 100, depth=3)
        self.assertEqual(5, move, msg="P2 Move should be 5")
        # self.assertEqual(100, score, msg="P2 Score should be 100")

    def test_alpha_beta_3(self):
        board = np.array([[1, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0],
                          [2, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 1],
                          [2, 2, 0, 0, 2, 2, 2],
                          [1, 1, 0, 2, 1, 1, 2]])
        game = Game(board=board, mark=2, config=config)
        move, score = alphabeta(game, time_left=lambda: 100, depth=3)
        self.assertEqual(3, move, msg="P2 Move should be 3")
        # self.assertEqual(100, score, msg="P2 Score should be 100")


if __name__ == '__main__':
    unittest.main()

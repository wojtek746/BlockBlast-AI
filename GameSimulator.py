import numpy as np
from random import randint
from typing import List, Tuple

class GameSimulator:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=bool)
        self.score = 0
        self.combo = 0
        self.is_cleared_line = False
        self.shapes = [
            # 1x1
            [[True, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            # 1x2
            [[True, True, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            # 1x3
            [[True, True, True, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            # 1x4
            [[True, True, True, True, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            # 1x5
            [[True, True, True, True, True],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #2x1
            [[True, False, False, False, False],
             [True, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #3x1
            [[True, False, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #4x1
            [[True, False, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False],
             [False, False, False, False, False]],
            #5x1
            [[True, False, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False]],
            # 2x2
            [[True, True, False, False, False],
             [True, True, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #`.
            [[True, False, False, False, False],
             [False, True, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #.`
            [[False, True, False, False, False],
             [True, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            # L-shapes
            [[True, True, True, False, False],
             [True, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            [[True, False, False, False, False],
             [True, True, True, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            [[True, True, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            [[True, True, False, False, False],
             [False, True, False, False, False],
             [False, True, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #2x3
            [[True, True, True, False, False],
             [True, True, True, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #3x2
            [[True, True, False, False, False],
             [True, True, False, False, False],
             [True, True, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #3x3
            [[True, True, True, False, False],
             [True, True, True, False, False],
             [True, True, True, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #E-shapes
            [[True, True, True, False, False],
             [False, True, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            [[False, True, False, False, False],
             [True, True, True, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            [[True, False, False, False, False],
             [True, True, False, False, False],
             [True, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            [[False, True, False, False, False],
             [True, True, False, False, False],
             [False, True, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
        ]
        self.shapes = [np.array(shape, dtype=bool) for shape in self.shapes]
        self.line_multipliers = [1, 2, 6, 12, 20]
        self.shop = [np.zeros((5, 5), dtype=bool), np.zeros((5, 5), dtype=bool), np.zeros((5, 5), dtype=bool)]

    def start(self):
        for i in range(8):
            for j in range(8):
                self.board[i][j] = randint(0, 1) == 1
        self.reload_shop()

        lines_cleared = self.clear_full_lines()

        if lines_cleared > 0:
            self.is_cleared_line = True
            self.combo = 1
            multiplier = self.line_multipliers[min(lines_cleared - 1, len(self.line_multipliers) - 1)]
            self.score += 10 * multiplier

    def reload_shop(self):
        for i in range(3):
            self.shop[i] = self.shapes[randint(0, len(self.shapes) - 1)] #potem przerobić, żeby było zależne od ilości pustych kafelków
        if self.is_cleared_line:
            self.is_cleared_line = False
        else:
            self.combo = 0

    def combo_points(self, n):
        if 0 < n < 6:
            return n * 10
        if 5 < n < 11:
            return n * 15
        if 10 < n:
            return n * 20
        return 0

    def can_place_shape(self, shape: List[List[bool]], row, col):
        for i in range(len(shape)):
            for j in range(len(shape[0])):
                if shape[i][j]:
                    if row + i >= 8 or col + j >= 8:
                        return False
                    if self.board[row + i][col + j]:
                        return False
        return True

    def place_shape(self, shop_index, row, col):
        if shop_index < 0 or shop_index >= 3:
            return False

        shape = self.shop[shop_index]
        if np.array_equal(shape, np.zeros((5, 5), dtype=bool)):
            return False

        if not self.can_place_shape(shape, row, col):
            return False

        for i in range(len(shape)):
            for j in range(len(shape[0])):
                if shape[i][j]:
                    self.board[row + i][col + j] = True

        lines_cleared = self.clear_full_lines()

        if lines_cleared > 0:
            self.is_cleared_line = True
            self.combo += 1
            multiplier = self.line_multipliers[min(lines_cleared - 1, len(self.line_multipliers) - 1)]
            self.score += self.combo_points(self.combo) * multiplier

        self.shop[shop_index] = np.zeros((5, 5), dtype=bool)
        if all(np.array_equal(s, np.zeros((5, 5), dtype=bool)) for s in self.shop):
            self.reload_shop()
        return True

    def clear_full_lines(self):
        lines_to_remove = []

        for i in range(8):
            if all(self.board[i]):
                lines_to_remove.append(i)

        for j in range(8):
            if all(self.board[:, j]):
                self.board[:, j] = False

        for i in lines_to_remove:
            self.board[i, :] = False

        return len(lines_to_remove)

    def get_valid_moves(self, shop_index):
        if shop_index < 0 or shop_index >= 3:
            return []
        shape = self.shop[shop_index]
        if not np.any(shape):
            return []
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self.can_place_shape(shape, row, col):
                    valid_moves.append((row, col))
        return valid_moves

    def get_all_valid_actions(self):
        actions = []
        for i in range(3):
            valid_moves = self.get_valid_moves(i)
            for row, col in valid_moves:
                actions.append((i, row, col))
        return actions

    def is_game_over(self):
        for i in range(3):
            if self.get_valid_moves(i):
                return False
        return True

    def get_state(self):
        return np.concatenate([
            self.board.flatten().astype(float),
            np.zeros(64, dtype=float), #dla przyszłych liczb na kafelkach
            self.shop[0].flatten().astype(float),
            self.shop[1].flatten().astype(float),
            self.shop[2].flatten().astype(float),
            [0 if self.combo == 0 else 1 - 1 / self.combo, 0 if self.is_cleared_line else 1],
        ])

    def copy(self):
        new_game = GameSimulator()
        new_game.board = self.board.copy()
        new_game.score = self.score
        new_game.combo = self.combo
        new_game.is_cleared_line = self.is_cleared_line
        new_game.shop = [s.copy() for s in self.shop]
        return new_game
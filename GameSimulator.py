import numpy as np
from random import randint, shuffle, choices, uniform, sample
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
            #2x1
            [[True, False, False, False, False],
             [True, False, False, False, False],
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
             [False, False, False, False, False]], #very small (5)
            # 2x2
            [[True, True, False, False, False],
             [True, True, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #3x1
            [[True, False, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #small L shapes
            [[True, True, False, False, False],
             [True, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            [[True, True, False, False, False],
             [False, True, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            [[True, False, False, False, False],
             [True, True, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            [[False, True, False, False, False],
             [True, True, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            # 1x3
            [[True, True, True, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]], #small (7)
            # 1x4
            [[True, True, True, True, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #4x1
            [[True, False, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False],
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
             [False, False, False, False, False]], #medium (10)
            #large L-shapes
            [[True, True, True, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            [[True, False, False, False, False],
             [True, False, False, False, False],
             [True, True, True, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            [[True, True, True, False, False],
             [False, False, True, False, False],
             [False, False, True, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            [[False, False, True, False, False],
             [False, False, True, False, False],
             [True, True, True, False, False],
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
            # 1x5
            [[True, True, True, True, True],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]],
            #5x1
            [[True, False, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False],
             [True, False, False, False, False]], #large (8)
            #3x3
            [[True, True, True, False, False],
             [True, True, True, False, False],
             [True, True, True, False, False],
             [False, False, False, False, False],
             [False, False, False, False, False]], #very large (1)
        ]
        self.shapes = [np.array(shape, dtype=bool) for shape in self.shapes]
        self.line_multipliers = [1, 2, 6, 12, 20]
        self.shop = [np.zeros((5, 5), dtype=bool), np.zeros((5, 5), dtype=bool), np.zeros((5, 5), dtype=bool)]

    def start(self, episode):
        if episode < 20000:
            ratio = 0.1 + (episode / 20000) * 0.1
        elif episode < 50000:
            ratio = 0.2 + (episode - 20000) / 30000 * 0.2
        elif episode < 80000:
            ratio = 0.4 + (episode - 50000) / 30000   * 0.2
        else:
            ratio = uniform(0.1, 0.7)

        target_cells = int(64 * 0.05)
        positions = [(i, j) for i in range(8) for j in range(8)]
        filled_positions = sample(positions, target_cells)
        for i, j in filled_positions:
            self.board[i][j] = True

        self.reload_shop()

        lines_cleared = self.clear_full_lines()

        if lines_cleared > 0:
            self.is_cleared_line = True
            self.combo = 1
            multiplier = self.line_multipliers[min(lines_cleared - 1, len(self.line_multipliers) - 1)]
            self.score += 10 * multiplier

    def reload_shop(self):
        self.shop = self.get_suitable_shapes()
        if self.is_cleared_line:
            self.is_cleared_line = False
        else:
            self.combo = 0

    def get_suitable_shapes(self):
        fill_ratio = (64 - np.sum(~self.board)) / 64

        very_small_shapes = self.shapes[1:5]
        small_shapes = self.shapes[5:12]
        medium_shapes = self.shapes[12:22]
        large_shapes = self.shapes[22:30]
        very_large_shapes = self.shapes[30:31]

        if fill_ratio > 0.9:
            suitable_shapes = very_small_shapes
        elif fill_ratio > 0.8:
            suitable_shapes = very_small_shapes + small_shapes
        elif fill_ratio > 0.6:
            suitable_shapes = very_small_shapes + small_shapes + medium_shapes
        elif fill_ratio > 0.3:
            suitable_shapes = very_small_shapes + small_shapes + medium_shapes + large_shapes
        else:
            suitable_shapes = very_small_shapes + small_shapes + medium_shapes + large_shapes + very_large_shapes

        def get_all_positions_for_shape(shape):
            positions = []
            for row in range(8):
                for col in range(8):
                    if self.can_place_shape(shape, row, col):
                        positions.append((row, col))
            return positions

        def positions_overlap(shape1, pos1, shape2, pos2):
            occupied1 = set()
            for i in range(len(shape1)):
                for j in range(len(shape1[0])):
                    if shape1[i][j]:
                        occupied1.add((pos1[0] + i, pos1[1] + j))

            for i in range(len(shape2)):
                for j in range(len(shape2[0])):
                    if shape2[i][j]:
                        if (pos2[0] + i, pos2[1] + j) in occupied1:
                            return True
            return False

        def has_non_overlapping_positions(shape1, positions1, shape2):
            positions2 = get_all_positions_for_shape(shape2)
            positions = []

            for pos2 in positions2:
                overlaps_with_any = False
                for pos1 in positions1:
                    if positions_overlap(shape1, pos1, shape2, pos2):
                        overlaps_with_any = True
                        break
                if not overlaps_with_any:
                    positions.append(pos2)
            return positions

        result_shapes = []

        max_attempts = 100
        first_shape = None
        first_positions = []
        weights = list(range(1, len(suitable_shapes) + 1))
        for attempt in range(max_attempts):
            candidate = choices(suitable_shapes, weights=weights)[0]
            positions = get_all_positions_for_shape(candidate)
            if positions:
                first_shape = candidate
                first_positions = positions
                result_shapes.append(candidate)
                break
        if not result_shapes:
            return [self.shapes[0], self.shapes[0], self.shapes[0]]

        second_shape = None
        second_positions = []
        for attempt in range(max_attempts):
            candidate = choices(suitable_shapes, weights=weights)[0]
            positions = has_non_overlapping_positions(first_shape, first_positions, candidate)
            if positions:
                second_shape = candidate
                second_positions = positions
                result_shapes.append(candidate)
                break
        if second_shape is None:
            return [result_shapes[0], self.shapes[0], self.shapes[0]]

        third_shape = None
        for attempt in range(max_attempts):
            candidate = choices(suitable_shapes, weights=weights)[0]
            pos1 = has_non_overlapping_positions(first_shape, first_positions, candidate)
            pos2 = has_non_overlapping_positions(second_shape, second_positions, candidate)
            found = False
            if pos1 and pos2:
                for p1 in pos1:
                    for p2 in pos2:
                        if p1 == p2:
                            third_shape = candidate
                            result_shapes.append(candidate)
                            found = True
                            break
                    if found:
                        break
            if found:
                break
        if third_shape is None:
            result_shapes.append(self.shapes[0])
        shuffle(result_shapes)

        return result_shapes

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

    def place_shape(self, shop_index, row, col, is_reload=True):
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
                    self.score += 1

        lines_cleared = self.clear_full_lines()

        if lines_cleared > 0:
            self.is_cleared_line = True
            self.combo += 1
            multiplier = self.line_multipliers[min(lines_cleared - 1, len(self.line_multipliers) - 1)]
            self.score += self.combo_points(self.combo) * multiplier

        self.shop[shop_index] = np.zeros((5, 5), dtype=bool)
        if is_reload and all(np.array_equal(s, np.zeros((5, 5), dtype=bool)) for s in self.shop):
            self.reload_shop()
        return (True, lines_cleared)

    def clear_full_lines(self):
        lines_to_remove = []
        number = 0

        for i in range(8):
            if all(self.board[i]):
                lines_to_remove.append(i)

        for j in range(8):
            if all(self.board[:, j]):
                self.board[:, j] = False
                number += 1

        for i in lines_to_remove:
            self.board[i, :] = False
            number += 1

        return number

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
                actions.append(i * 64 + row * 8 + col)
        return actions

    def is_game_over(self):
        for i in range(3):
            if self.get_valid_moves(i):
                return False
        return True

    def get_state(self):
        #funkcja wysyła stan gry do AI
        row_fullness = [np.sum(self.board[i]) for i in range(8)]
        col_fullness = [np.sum(self.board[:, j]) for j in range(8)]
        valid = self.get_all_valid_actions()
        return np.concatenate([
            self.board.flatten().astype(float),  # 64 (plansza 8x8)
            [8 - np.sum(self.board[i]) for i in range(8)],  # 8 (liczba pustych kafelków w poziomie)
            [8 - np.sum(self.board[:, j]) for j in range(8)],  # 8 (liczba pustych kafelków w pionie)
            [1 if sum >= 6 else 0 for sum in row_fullness],  # 8 (czy wiersz ma przynajmniej 6 kafelków)
            [1 if sum >= 6 else 0 for sum in col_fullness],  # 8 (czy kolumna ma przynajmniej 6 kafelków)
            self.shop[0].flatten().astype(float),  # 25 (sklep 5x5)
            self.shop[1].flatten().astype(float),  # 25 (sklep 5x5)
            self.shop[2].flatten().astype(float),  # 25 (sklep 5x5)
        ])

    def copy(self):
        new_game = GameSimulator()
        new_game.board = self.board.copy()
        new_game.score = self.score
        new_game.combo = self.combo
        new_game.is_cleared_line = self.is_cleared_line
        new_game.shop = [s.copy() for s in self.shop]
        return new_game
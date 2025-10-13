import copy
from random import choice

from plansza2 import predicted_reward

best = 0

def save(reward):
    global best
    if reward > best:
        best = reward
        with open("best_reward.txt", "w") as f:
            f.write(str(reward))

def load():
    global best
    with open("best_reward.txt", "r") as f:
        best = int(f.read())

def get_valid_moves(board, shape):
    if not shape:
        return []
    valid_moves = []
    for row in range(8):
        for col in range(8):
            place = True
            for i in range(5):
                for j in range(5):
                    if shape[i][j]:
                        if row + i >= 8 or col + j >= 8:
                            place = False
                            break
                        if board[row + i][col + j]:
                            place = False
                            break
                if not place:
                    break
            if place:
                valid_moves.append((row, col))
    return valid_moves

def get_all_valid_actions(board, shop):
    actions = []
    for i in range(3):
        valid_moves = get_valid_moves(board, shop[i])
        for row, col in valid_moves:
            actions.append(i * 64 + row * 8 + col)
    return actions

def simulate_place_shape(board, shop, action):
    shop_index, row, col = action // 64, (action % 64) // 8, action % 8
    shape = shop[shop_index]
    if not shape:
        print("sklep pusty")

    new_board = copy.deepcopy(board)
    for i in range(5):
        for j in range(5):
            if shape[i][j]:
                new_board[row + i][col + j] = True

    lines_to_remove = []
    for i in range(8):
        if all(new_board[i]):
            lines_to_remove.append(i)
    for j in range(8):
        col_full = True
        for i in range(8):
            if not new_board[i][j]:
                col_full = False
                break
        if col_full:
            lines_to_remove.append(j)
    shop[shop_index] = None
    return len(lines_to_remove), shop, new_board

def best_action(board, shop, valid_actions):
    best = []
    best_reward = -10000
    best_lines_cleared = 0
    for action in valid_actions:
        lines_cleared, new_shop, new_board = simulate_place_shape(board, copy.deepcopy(shop), action)
        reward = predicted_reward(new_board, lines_cleared, new_shop)
        if reward > best_reward:
            best = [action]
            best_reward = reward
            best_lines_cleared = lines_cleared
        elif reward == best_reward:
            best.append(action)
            if lines_cleared > best_lines_cleared:
                best_lines_cleared = lines_cleared
    save(best_reward)
    print("choice", best_reward, len(best), len(valid_actions))
    return choice(best), best_lines_cleared

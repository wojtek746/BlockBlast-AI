import copy
import time
import subprocess
import cv2

from plansza2 import predicted_reward, penalty_for_losing, clear_lines
from random import choice

def tap(x, y):
    subprocess.run(["adb", "shell", "input", "tap", str(x), str(y)])

def swipe(x1, y1, x2, y2, duration_ms=300):
    subprocess.run(["adb", "shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])

def get_screenshot(filename='screen.png'):
    subprocess.run(["adb", "shell", "screencap", "-p", "/sdcard/screen.png"])
    subprocess.run(["adb", "pull", "/sdcard/screen.png", filename], stderr=subprocess.DEVNULL)
    subprocess.run(["adb", "shell", "rm", "/sdcard/screen.png"])

def move(shop, action):
    t = 1.2
    shop_index, row, col = action // 64, (action % 64) // 8, action % 8
    match shop_index:
        case 0:
            from_x = 225
            dx = -116
        case 1:
            from_x = 540
            dx = -343
        case 2:
            from_x = 855
            dx = -568
    from_y = 1849
    dy = 690

    shape = shop[shop_index]
    x = -1
    y = -1
    for i in range(5):
        for j in range(5):
            if shape[i][j]:
                if i > y:
                    y = i
                if j > x:
                    x = j
    x += 1
    y += 1

    dx += x * 43
    dy -= y * 43

    dx += col * 86
    dy -= row * 86
    #print(row, col)

    swipe(from_x, from_y, from_x + dx, from_y - dy, int(t*(dx**2+dy**2)**0.5)+10)

blue = [(148, 81, 58), (148, 85, 58), (148, 85, 66), (140, 81, 58), (140, 73, 49), (132, 69, 49), (140, 77, 58), (132, 65, 41), (132, 65, 49), (140, 77, 49), (156, 85, 66)]

def isin(board, x, y):
    global blue
    return tuple(board[y, x]) in blue

def array_of_shape(b, n):
    global blue
    match(n):
        case 0:
            x = 198
        case 1:
            x = 513
        case 2:
            x = 828
        case _:
            return None
    y = 1822

    if (isin(b, x+5, y+5) and isin(b, x+49, y+5) and isin(b, x+5, y+49) and isin(b, x+49, y+49) and
            isin(b, x-5, y+5) and isin(b, x-5, y+49) and isin(b, x+59, y+5) and isin(b, x+59, y+49) and
            isin(b, x+5, y-5) and isin(b, x+49, y-5) and isin(b, x+5, y+59) and isin(b, x+49, y+59)):
        return None

    if isin(b, x-5, y+5) and isin(b, x-5, y+49) and isin(b, x+59, y+5) and isin(b, x+59, y+49) and isin(b, x-5, y-5) and isin(b, x+59, y-5): #1xX
        if isin(b, x+5, y-5) and isin(b, x+49, y-5) and isin(b, x+5, y+59) and isin(b, x+49, y+59):
            return [[True, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-32) and isin(b, x+49, y-32) and isin(b, x+5, y+86) and isin(b, x+49, y+86):
            return [[True, False, False, False, False], [True, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-59) and isin(b, x+49, y-59) and isin(b, x+5, y+113) and isin(b, x+49, y+113):
            return [[True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-86) and isin(b, x+49, y-86) and isin(b, x+5, y+140) and isin(b, x+49, y+140):
            return [[True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-113) and isin(b, x+49, y-113) and isin(b, x+5, y+167) and isin(b, x+49, y+167):
            return [[True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False], [True, False, False, False, False]]
    if isin(b, x-32, y+5) and isin(b, x-32, y+49) and isin(b, x+86, y+5) and isin(b, x+86, y+49) and isin(b, x-32, y-5) and isin(b, x+86, y-5): #2xX
        if isin(b, x+5, y-5) and isin(b, x+49, y-5) and isin(b, x+5, y+59) and isin(b, x+49, y+59):
            return [[True, True, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-32) and isin(b, x+49, y-32) and isin(b, x+5, y+86) and isin(b, x+49, y+86):
            return [[not isin(b, x, y), not isin(b, x+54, y), False, False, False], [not isin(b, x, y+54), not isin(b, x+54, y+54), False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-59) and isin(b, x+49, y-59) and isin(b, x+5, y+113) and isin(b, x+49, y+113):
            return [[not isin(b, x, y-27), not isin(b, x+54, y-27), False, False, False], [not isin(b, x, y+27), not isin(b, x+54, y+27), False, False, False], [not isin(b, x, y+81), not isin(b, x+54, y+81), False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-86) and isin(b, x+49, y-86) and isin(b, x+5, y+140) and isin(b, x+49, y+140):
            return [[not isin(b, x, y-54), not isin(b, x+54, y-54), False, False, False], [not isin(b, x, y), not isin(b, x+54, y), False, False, False], [not isin(b, x, y+54), not(b, x+54, y+54), False, False, False], [not isin(b, x, y+108), not isin(b, x+54, y+108), False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-113) and isin(b, x+49, y-113) and isin(b, x+5, y+167) and isin(b, x+49, y+167):
            return [[not isin(b, x, y-81), not isin(b, x+54, y-81), False, False, False], [not isin(b, x, y-27), not isin(b, x+54, y-27), False, False, False], [not isin(b, x, y+27), not(b, x+54, y+27), False, False, False], [not isin(b, x, y+81), not isin(b, x+54, y+81), False, False, False], [not isin(b, x, y+135), not isin(b, x+54, y+135), False, False, False]]
    if isin(b, x-59, y+5) and isin(b, x-59, y+49) and isin(b, x+113, y+5) and isin(b, x+113, y+49): #3xX
        if isin(b, x+5, y-5) and isin(b, x+49, y-5) and isin(b, x+5, y+59) and isin(b, x+49, y+59) and isin(b, x-5, y-5) and isin(b, x-5, y+59):
            return [[True, True, True, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-32) and isin(b, x+49, y-32) and isin(b, x+5, y+86) and isin(b, x+49, y+86) and isin(b, x-32, y-32) and isin(b, x+86, y-32):
            return [[not isin(b, x-27, y), not isin(b, x+27, y), not isin(b, x+81, y), False, False], [not isin(b, x-27, y+54), not isin(b, x+27, y+54), not isin(b, x+81, y+54), False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-59) and isin(b, x+49, y-59) and isin(b, x+5, y+113) and isin(b, x+49, y+113):
            return [[not isin(b, x-27, y-27), not isin(b, x+27, y-27), not isin(b, x+81, y-27), False, False], [not isin(b, x-27, y+27), not isin(b, x+27, y+27), not isin(b, x+81, y+27), False, False], [not isin(b, x-27, y+81), not isin(b, x+27, y+81), not isin(b, x+81, y+81), False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-86) and isin(b, x+49, y-86) and isin(b, x+5, y+140) and isin(b, x+49, y+140):
            return [[not isin(b, x-27, y-54), not isin(b, x+27, y-54), not isin(b, x+81, y-54), False, False], [not isin(b, x-27, y), not isin(b, x+27, y), not isin(b, x+81, y), False, False], [not isin(b, x-27, y+54), not isin(b, x+27, y+54), not isin(b, x+81, y+54), False, False], [not isin(b, x-27, y+108), not isin(b, x+27, y+108), not isin(b, x+81, y+108), False, False], [False, False, False, False, False]]
    if isin(b, x-86, y+5) and isin(b, x-86, y+49) and isin(b, x+140, y+5) and isin(b, x+140, y+49): #4xX
        if isin(b, x+5, y-5) and isin(b, x+49, y-5) and isin(b, x+5, y+59) and isin(b, x+49, y+59):
            return [[True, True, True, True, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-32) and isin(b, x+49, y-32) and isin(b, x+5, y+86) and isin(b, x+49, y+86):
            return [[not isin(b, x-54, y), not isin(b, x, y), not isin(b, x+54, y), not isin(b, x+108, y), False], [not isin(b, x-54, y+54), not isin(b, x, y+54), not isin(b, x+54, y+54), not isin(b, x+108, y+54), False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-59) and isin(b, x+49, y-59) and isin(b, x+5, y+113) and isin(b, x+49, y+113):
            return [[not isin(b, x-54, y-27), not isin(b, x, y-27), not isin(b, x+54, y-27), not isin(b, x+108, y-27), False], [not isin(b, x-54, y+27), not isin(b, x, y+27), not isin(b, x+54, y+27), not isin(b, x+108, y+27), False], [not isin(b, x-54, y+81), not isin(b, x, y+81), not isin(b, x+54, y+81), not isin(b, x+108, y+81), False], [False, False, False, False, False], [False, False, False, False, False]]
    if isin(b, x-113, y+5) and isin(b, x-113, y+49) and isin(b, x+167, y+5) and isin(b, x+167, y+49): #5xX
        if isin(b, x+5, y-5) and isin(b, x+49, y-5) and isin(b, x+5, y+59) and isin(b, x+49, y+59):
            return [[True, True, True, True, True], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]
        if isin(b, x+5, y-32) and isin(b, x+49, y-32) and isin(b, x+5, y+86) and isin(b, x+49, y+86):
            return [[not isin(b, x-81, y), not isin(b, x-27, y), not isin(b, x+27, y), not isin(b, x+81, y), not isin(b, x+135, y)], [not isin(b, x-81, y+54), not isin(b, x-27, y+54), not isin(b, x+27, y+54), not isin(b, x+81, y+54), not isin(b, x+135, y+54)], [False, False, False, False, False], [False, False, False, False, False], [False, False, False, False, False]]

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

def board_to_bool_array(board):
    start_x = 120
    start_y = 670
    blanks = [(25, 36, 66), (66, 61, 123), (66, 36, 25)]
    result = []
    for i in range(8):
        row = []
        for j in range(8):
            x = start_x + j * 120
            y = start_y + i * 120
            row.append(not tuple(board[y, x]) in blanks)
        result.append(row)
    return result

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

def step():
    get_screenshot()
    screen = cv2.imread('screen.png')
    board = board_to_bool_array(screen)
    shop = []
    for i in range(3):
        shop.append(array_of_shape(screen, i))
    valid_actions = get_all_valid_actions(board, shop)
    if not valid_actions:
        if all(all(row) for row in board):
            print("plansza nie istnieje")
            tap(1000, 230)
            time.sleep(0.2)
            tap(500, 1700)
            return 0
        else:
            print("brak akcji")
            return 1

    action, lines_cleared = best_action(board, shop, valid_actions)
    move(shop, action)

    return lines_cleared

def main():
    load()
    while True:
        lines_cleared = step()
        if lines_cleared > 0:
            time.sleep(1.2)
        time.sleep(0.3)

if __name__ == "__main__":
    main()
    #dekompilator java, żeby ogarnąć reload_shop()
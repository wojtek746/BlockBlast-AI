# Author           : Piotr Raczek
# Created On       : 25.09.2025
# Last Modified By : Piotr Raczek
# Last Modified On : 26.09.2025
import copy

# configuration variables
penalty_for_losing = -1000


# it is my static variable in Python -PR-
ocena_planszy = 0


# „predicted_reward” — ocena planszy (8x8)-T/F -PR-
def predicted_reward(board, lines_cleared, shop, isWithoutOcenaPlanszy = False):
    global ocena_planszy # static -PR-
    nowa_ocena_planszy = 0
    T = [[1] * 10 for _ in range(10)]
    ki, kj = [0] * 10, [0] * 10 # kubełek i oraz kubełek j -PR-

    for i in range(8):
        for j in range(8):
            T[i+1][j+1] = int(board[i][j]) # kopiowanie -PR-

    for i in range(1, 9):
        for j in range(1, 9):
            neighbours = T[i+1][j]+T[i-1][j]+T[i][j+1]+T[i][j-1]
            if T[i][j] == True:
                nowa_ocena_planszy += [0, 0, 2, 3, 5, 10][neighbours] # to do
            else:
                neighbours += T[i+1][j+1]+T[i+1][j-1]+T[i-1][j+1]+T[i-1][j-1]
                nowa_ocena_planszy += [  0, -1, -3, -5, -9, -14, -20, -27, -35][neighbours] # to do

    ocena = nowa_ocena_planszy + 500 * lines_cleared
    if not isWithoutOcenaPlanszy:
        ocena -= ocena_planszy
        ocena_planszy = nowa_ocena_planszy
    return (ocena + 50) * 10 + is_imposible_to_survive(board, shop) * penalty_for_losing

def fits_on(board, shape, i, j):
    for x in range(5):
        for y in range(5):
            if shape[x][y]:
                if i + x >= 8 or j + y >= 8:
                    return False
                if board[i + x][j + y]:
                    return False
    return True

def clear_lines(board):
    lines_to_remove = []
    for i in range(8):
        if all(board[i]):
            lines_to_remove.append(i)
    for j in range(8):
        col_full = True
        for i in range(8):
            if not board[i][j]:
                col_full = False
                break
        if col_full:
            for i in range(8):
                board[i][j] = False
    for i in lines_to_remove:
        for j in range(8):
            board[i][j] = False

def is_imposible_to_survive(board, shop):
    if not shop[0]:
        if not shop[1]:
            if not shop[2]:
                return 0
            for i in range(8):
                for j in range(8):
                    if fits_on(board, shop[2], i, j):
                        return 0
            return 1
        for i in range(8):
            for j in range(8):
                if fits_on(board, shop[1], i, j):
                    if not shop[2]:
                        return 0
                    b = copy.deepcopy(board)
                    for x in range(5):
                        for y in range(5):
                            if shop[1][x][y]:
                                b[i+x][j+y] = True
                    clear_lines(b)
                    for i2 in range(8):
                        for j2 in range(8):
                            if fits_on(b, shop[2], i2, j2):
                                return 0
        return 1
    for i in range(8):
        for j in range(8):
            if fits_on(board, shop[0], i, j):
                b = copy.deepcopy(board)
                for x in range(5):
                    for y in range(5):
                        if shop[0][x][y]:
                            b[i+x][j+y] = True
                clear_lines(b)
                if not shop[1]:
                    if not shop[2]:
                        return 0
                    for i2 in range(8):
                        for j2 in range(8):
                            if fits_on(b, shop[2], i2, j2):
                                return 0
                else:
                    for i1 in range(8):
                        for j1 in range(8):
                            if fits_on(b, shop[1], i1, j1):
                                if not shop[2]:
                                    return 0
                                b2 = copy.deepcopy(b)
                                for x in range(5):
                                    for y in range(5):
                                        if shop[1][x][y]:
                                            b2[i1+x][j1+y] = True
                                clear_lines(b2)
                                for i2 in range(8):
                                    for j2 in range(8):
                                        if fits_on(b2, shop[2], i2, j2):
                                            return 0
    return 1
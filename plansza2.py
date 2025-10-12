# Author           : Piotr Raczek
# Created On       : 25.09.2025
# Last Modified By : Piotr Raczek
# Last Modified On : 26.09.2025
import copy

# configuration variables
penalty_for_losing = -5000


# it is my static variable in Python -PR-


# „predicted_reward” — ocena planszy (8x8)-T/F -PR-
def predicted_reward(board, lines_cleared, shop, isWithoutOcenaPlanszy = False):
    global ocena_planszy # static -PR-
    nowa_ocena_planszy = 1000

    board = clear_lines(board)

    T = [[1] * 10 for _ in range(10)]
    #ki, kj = [0] * 10, [0] * 10 # kubełek i oraz kubełek j -PR-

    for i in range(8):
        for j in range(8):
            T[i+1][j+1] = int(board[i][j]) # kopiowanie -PR-

    for i in range(1, 9):
        for j in range(1, 9):
            neighbours = T[i+1][j]+T[i-1][j]+T[i][j+1]+T[i][j-1]
            middle = int(((i - 4.5) ** 2 + (j - 4.5) ** 2) // 2)
            if T[i][j] == True:
                nowa_ocena_planszy += 5 + middle
                nowa_ocena_planszy += [-9,-3, 0,  5,  20][neighbours]  # to do
            else:
                nowa_ocena_planszy += [10, 0,-5,-25,-125][neighbours] # to do
    #print(nowa_ocena_planszy + 1000 * lines_cleared)
    if is_imposible_to_survive(copy.deepcopy(board), shop):
        print("Imposible! ", end="")
        nowa_ocena_planszy += penalty_for_losing
    return nowa_ocena_planszy + 1000 * lines_cleared

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
    return board

def is_imposible_to_survive(board, shop):
    elementy = []
    if shop[0]:
        elementy.append(shop[0])
    if shop[1]:
        elementy.append(shop[1])
    if shop[2]:
        elementy.append(shop[2])
    #print(len(elementy), end="")
    if len(elementy) == 2: #else = 1
        for ie in range(len(elementy)):
            e = elementy[ie]
            for i in range(8):
                for j in range(8):
                    if fits_on(board, e, i, j):
                        b = copy.deepcopy(board)
                        for x in range(5):
                            for y in range(5):
                                if e[x][y]:
                                    b[i+x][j+y] = True
                        b = clear_lines(b)
                        for je in range(len(elementy)):
                            if je == ie:
                                continue
                            e2 = elementy[je]
                            for i2 in range(8):
                                for j2 in range(8):
                                    if fits_on(b, e2, i2, j2):
                                        return 0
    elif len(elementy) == 1:
        for i in range(8):
            for j in range(8):
                if fits_on(board, elementy[0], i, j):
                    return 0
    else:
        return 0
    return 1
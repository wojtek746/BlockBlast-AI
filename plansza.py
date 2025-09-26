# Author           : Piotr Raczek
# Created On       : 25.09.2025
# Last Modified By : Piotr Raczek
# Last Modified On : 26.09.2025


# configuration variables
penalty_for_losing = -1000


# it is my static variable in Python -PR-
ocena_planszy = 0


# „predicted_reward” — ocena planszy (8x8)-T/F -PR-
def predicted_reward(board, lines_cleared, ocena_planszy=ocena_planszy):
	#global ocena_planszy # static -PR-
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

	ocena = nowa_ocena_planszy - ocena_planszy + 500 * lines_cleared
	ocena_planszy = nowa_ocena_planszy

	ocena = nowa_ocena_planszy - ocena_planszy + 500 * lines_cleared
	ocena_planszy = nowa_ocena_planszy
	return (ocena + 50) * 10

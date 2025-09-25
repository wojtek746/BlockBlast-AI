ocena_planszy = 0

def predicted_reward(state):
    global ocena_planszy
    T = state.board
    return ocena
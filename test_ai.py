import torch
import numpy as np
from GameSimulator import GameSimulator
from GameAI import DQN

def print_board(board, title="Plansza"):
    print(f"\n{title}:")
    print("  " + " ".join([str(i) for i in range(8)]))
    for i, row in enumerate(board):
        row_str = str(i) + " "
        for cell in row:
            row_str += "█ " if cell else "· "
        print(row_str)

def print_shape(shape, title="Kształt"):
    print(f"\n{title}:")
    for row in shape:
        row_str = ""
        for cell in row:
            row_str += "█ " if cell else "· "
        print(row_str)

def print_shop(shop):
    print("\nSklep:")
    for i, shape in enumerate(shop):
        print(f"Pozycja {i}:")
        if np.any(shape):
            for row in shape:
                row_str = ""
                for cell in row:
                    row_str += "█ " if cell else "· "
                print(row_str)
        else:
            print("(pusty)")
        print()

def action_to_readable(action):
    shop_index = action // 64
    row = (action % 64) // 8
    col = action % 8
    return shop_index, row, col


def test_trained_ai(model_path='trained_model.pt', num_games=1, detailed=True):
    # Wczytaj model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = DQN().to(device)

    try:
        q_network.load_state_dict(torch.load(model_path, map_location=device))
        q_network.eval()
        print(f"Wczytano model z {model_path}")
    except FileNotFoundError:
        print(f"Nie znaleziono pliku {model_path}")
        return

    def ai_act(state, valid_actions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = q_network(state_tensor)

        # Maskowanie nieprawidłowych akcji
        masked_q_values = q_values.clone()
        valid_set = set(valid_actions)
        for i in range(192):
            if i not in valid_set:
                masked_q_values[0][i] = float('-inf')

        return masked_q_values.argmax().item()

    all_scores = []

    for game_num in range(num_games):
        print(f"\n{'=' * 60}")
        print(f"GRA {game_num + 1}")
        print(f"{'=' * 60}")

        game = GameSimulator()
        game.start(0)  # Rozpocznij z episode=0

        move_count = 0

        if detailed:
            print_board(game.board, "Stan początkowy planszy")
            print_shop(game.shop)
            print(f"Punkty początkowe: {game.score}")

        while not game.is_game_over():
            valid_actions = game.get_all_valid_actions()

            if not valid_actions:
                print("Brak dostępnych ruchów!")
                break

            move_count += 1

            if detailed:
                print(f"\n{'-' * 40}")
                print(f"RUCH {move_count}")
                print(f"{'-' * 40}")
                print(f"Dostępne akcje: {len(valid_actions)}")

            state = game.get_state()
            action = ai_act(state, valid_actions)

            shop_index, row, col = action_to_readable(action)

            if detailed:
                print(f"AI wybiera: Kształt {shop_index}, pozycja ({row}, {col})")
                print_shape(game.shop[shop_index], f"Wybrany kształt (sklep {shop_index})")

            old_score = game.score
            old_combo = game.combo
            old_board = game.board.copy()

            success = game.place_shape(shop_index, row, col)

            if not success:
                print("BŁĄD: Nie udało się wykonać ruchu!")
                break

            points_gained = game.score - old_score

            if detailed:
                print_board(game.board, "Plansza po ruchu")
                print(f"Punkty zdobyte: {points_gained}")
                print(f"Całkowite punkty: {game.score}")
                print(f"Combo: {old_combo} -> {game.combo}")
                if game.is_cleared_line:
                    print("🔥 WYCZYSZCZONO LINIE!")

                # Sprawdź czy sklep się odnowił
                if all(np.array_equal(s, np.zeros((5, 5), dtype=bool)) for s in game.shop):
                    print("Sklep pusty - odnowienie...")
                else:
                    print_shop(game.shop)

        final_score = game.score
        all_scores.append(final_score)

        print(f"\n{'=' * 40}")
        print(f"KONIEC GRY {game_num + 1}")
        print(f"{'=' * 40}")
        print(f"Liczba ruchów: {move_count}")
        print(f"Końcowy wynik: {final_score}")
        print_board(game.board, "Końcowy stan planszy")

        # Sprawdź, dlaczego gra się skończyła
        print("\nAnaliza końca gry:")
        for i in range(3):
            valid_moves = game.get_valid_moves(i)
            print(f"Kształt {i}: {len(valid_moves)} możliwych pozycji")
            if len(valid_moves) == 0 and np.any(game.shop[i]):
                print_shape(game.shop[i], f"Niemożliwy do umieszczenia kształt {i}")

    print(f"\n{'=' * 60}")
    print("PODSUMOWANIE")
    print(f"{'=' * 60}")
    print(f"Liczba gier: {num_games}")
    print(f"Średni wynik: {np.mean(all_scores):.1f}")
    print(f"Najlepszy wynik: {max(all_scores)}")
    print(f"Najgorszy wynik: {min(all_scores)}")
    print(f"Odchylenie standardowe: {np.std(all_scores):.1f}")

    return all_scores

def test_single_detailed(model_path='trained_model.pt'):
    return test_trained_ai(model_path, 1, detailed=True)

if __name__ == "__main__":
    scores = test_single_detailed()
    print(scores)
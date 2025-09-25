import pygame
import sys
from GameSimulator import GameSimulator
import numpy as np

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BLUE = (0, 100, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)

BOARD_SIZE = 8
CELL_SIZE = 60
BOARD_WIDTH = BOARD_SIZE * CELL_SIZE
BOARD_HEIGHT = BOARD_SIZE * CELL_SIZE
SHOP_HEIGHT = 150
WINDOW_WIDTH = BOARD_WIDTH + 300  # Dodatkowe miejsce na info
WINDOW_HEIGHT = BOARD_HEIGHT + SHOP_HEIGHT + 100


class HumanPlayer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Block Blast - Human Player")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self.game = GameSimulator()
        self.game.start(0)

        self.selected_shop_index = None
        self.running = True

    def draw_board(self):
        start_x, start_y = 10, 10

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x = start_x + col * CELL_SIZE
                y = start_y + row * CELL_SIZE

                if self.game.board[row][col]:
                    color = BLUE
                else:
                    color = WHITE

                pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 2)

    def draw_shape(self, shape, start_x, start_y, cell_size=30, color=GREEN):
        for i in range(5):
            for j in range(5):
                if shape[i][j]:
                    x = start_x + j * cell_size
                    y = start_y + i * cell_size
                    pygame.draw.rect(self.screen, color, (x, y, cell_size, cell_size))
                    pygame.draw.rect(self.screen, BLACK, (x, y, cell_size, cell_size), 1)

    def draw_shop(self):
        shop_y = BOARD_HEIGHT + 20

        for i in range(3):
            shop_x = 10 + i * 160

            if self.selected_shop_index == i:
                color = YELLOW
            elif np.any(self.game.shop[i]):
                color = GRAY
            else:
                color = RED

            pygame.draw.rect(self.screen, color, (shop_x, shop_y, 150, 120))
            pygame.draw.rect(self.screen, BLACK, (shop_x, shop_y, 150, 120), 3)

            if np.any(self.game.shop[i]):
                self.draw_shape(self.game.shop[i], shop_x + 10, shop_y + 10)

    def draw_info(self):
        info_x = BOARD_WIDTH + 20

        score_text = self.font.render(f"Score: {self.game.score}", True, BLACK)
        self.screen.blit(score_text, (info_x, 20))

        combo_text = self.font.render(f"Combo: {self.game.combo}", True, BLACK)
        self.screen.blit(combo_text, (info_x, 60))

        instructions = [
            "Instrukcje:",
            "1. Kliknij kształt w sklepie",
            "2. Kliknij na planszę",
            "3. Spróbuj czyścić linie!",
            "",
            "R - Restart gry",
            "ESC - Wyjście"
        ]

        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, BLACK)
            self.screen.blit(text, (info_x, 120 + i * 25))

        if self.game.is_game_over():
            game_over_text = self.font.render("GAME OVER!", True, RED)
            self.screen.blit(game_over_text, (info_x, 350))

        valid_actions = self.game.get_all_valid_actions()
        moves_text = self.small_font.render(f"Dostępne ruchy: {len(valid_actions)}", True, BLACK)
        self.screen.blit(moves_text, (info_x, 380))

    def get_board_click(self, mouse_pos):
        x, y = mouse_pos
        if 10 <= x <= 10 + BOARD_WIDTH and 10 <= y <= 10 + BOARD_HEIGHT:
            col = (x - 10) // CELL_SIZE
            row = (y - 10) // CELL_SIZE
            return row, col
        return None

    def get_shop_click(self, mouse_pos):
        x, y = mouse_pos
        shop_y = BOARD_HEIGHT + 20

        if shop_y <= y <= shop_y + 120:
            for i in range(3):
                shop_x = 10 + i * 160
                if shop_x <= x <= shop_x + 150:
                    return i
        return None

    def handle_click(self, mouse_pos):
        shop_click = self.get_shop_click(mouse_pos)
        if shop_click is not None:
            if np.any(self.game.shop[shop_click]):
                self.selected_shop_index = shop_click
                print(f"Wybrano kształt {shop_click + 1}")
            return

        board_click = self.get_board_click(mouse_pos)
        if board_click is not None and self.selected_shop_index is not None:
            row, col = board_click
            success = self.game.place_shape(self.selected_shop_index, row, col)
            if success:
                print(f"Umieszczono kształt na pozycji ({row}, {col})")
                print(f"Nowy wynik: {self.game.score}")
                self.selected_shop_index = None
            else:
                print("Nie można umieścić kształtu w tym miejscu!")

    def restart_game(self):
        self.game = GameSimulator()
        self.game.start(0)
        self.selected_shop_index = None
        print("Gra zrestartowana!")

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_r:
                        self.restart_game()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Lewy przycisk myszy
                        self.handle_click(event.pos)

            self.screen.fill(WHITE)
            self.draw_board()
            self.draw_shop()
            self.draw_info()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = HumanPlayer()
    game.run()
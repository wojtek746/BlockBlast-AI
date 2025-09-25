import math
import random
import time
from GameSimulator import GameSimulator
import numpy as np

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # GameSimulator object
        self.parent = parent
        self.action = action  # Akcja która doprowadziła do tego stanu

        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = state.get_all_valid_actions().copy()
        self.is_terminal = state.is_game_over()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.total_reward / child.visits) +
            c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        if not self.untried_actions:
            return None

        action = self.untried_actions.pop()

        new_state = self.state.copy()
        shop_index = action // 64
        row = (action % 64) // 8
        col = action % 8

        success = new_state.place_shape(shop_index, row, col)
        if not success: # always False
            if self.untried_actions:
                return self.expand()
            return None

        child_node = MCTSNode(new_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def evaluate_action_with_uncertainty(self, simulations=10, max_moves=200):
        total_score = 0

        for _ in range(simulations):
            total_score += self.single_rollout(max_moves)

        return total_score / simulations

    def single_rollout(self, max_moves=200):
        current_state = self.state.copy()
        moves = 0

        while not current_state.is_game_over() and moves < max_moves:
            valid_actions = current_state.get_all_valid_actions()
            if not valid_actions:
                break

            action = self.rollout_policy(current_state, valid_actions)

            shop_index = action // 64
            row = (action % 64) // 8
            col = action % 8

            success = current_state.place_shape(shop_index, row, col)
            if not success:
                break
            moves += 1

        return current_state.score

    def rollout_policy(self, state, valid_actions):
        best_action = None
        best_score = -1

        for action in valid_actions:
            score = self.evaluate_action_with_uncertainty(state, action)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action else random.choice(valid_actions)

    def backpropagate(self, reward):
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)


class MCTS:
    def __init__(self, c_param=1.4, time_limit=5.0):
        self.c_param = c_param
        self.time_limit = time_limit

    def search(self, root_state):
        root = MCTSNode(root_state)

        start_time = time.time()
        iterations = 0

        while time.time() - start_time < self.time_limit:
            node = self.tree_policy(root)
            if node is None:
                break

            reward = node.rollout()
            node.backpropagate(reward)
            iterations += 1

        print(f"MCTS iterations: {iterations}, time: {time.time() - start_time:.2f}s")

        if not root.children:
            valid_actions = root_state.get_all_valid_actions()
            return random.choice(valid_actions) if valid_actions else None

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def tree_policy(self, node):
        while not node.is_terminal:
            if not node.is_fully_expanded():
                return node.expand()
            else:
                if not node.children:
                    return None
                node = node.best_child(self.c_param)
        return node
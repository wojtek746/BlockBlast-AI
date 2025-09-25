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

    def rollout(self, state, action, depth, max_depth, max_moves=5):
        current_state = state.copy()
        moves = 0

        shop_index = action // 64
        row = (action % 64) // 8
        col = action % 8

        success = current_state.place_shape(shop_index, row, col)
        if not success:
            return -1
        moves += 1

        if current_state.is_game_over():
            return -1

        while not current_state.is_game_over() and moves < max_moves:
            valid_actions = current_state.get_all_valid_actions()
            if not valid_actions:
                break

            action = self.rollout_policy(current_state, valid_actions, depth, max_depth)

            shop_index = action // 64
            row = (action % 64) // 8
            col = action % 8

            success = current_state.place_shape(shop_index, row, col)
            if not success:
                break
            moves += 1

        return current_state.score

    def simple_rollout_policy(self, state, valid_actions):
        best_actions = []
        best_score_increase = -1

        for action in valid_actions:
            test_state = state.copy()
            shop_index = action // 64
            row = (action % 64) // 8
            col = action % 8

            old_score = test_state.score
            success = test_state.place_shape(shop_index, row, col)

            if success:
                score_increase = test_state.score - old_score
                if score_increase > best_score_increase:
                    best_score_increase = score_increase
                    best_actions = [action]
                elif score_increase == best_score_increase:
                    best_actions.append(action)

        if best_actions and best_score_increase > 1:
            return random.choice(best_actions)

        return random.choice(valid_actions)

    def evaluate_action_with_uncertainty(self, state, action, depth, max_depth, simulations=3):
        total_score = 0

        for _ in range(simulations):
            total_score += self.rollout(state, action, depth, max_depth)

        return total_score / simulations

    def single_rollout(self, max_moves=10):
        current_state = self.state.copy()
        moves = 0

        if current_state.is_game_over():
            return -1

        while not current_state.is_game_over() and moves < max_moves:
            valid_actions = current_state.get_all_valid_actions()
            if not valid_actions:
                break

            action = self.rollout_policy(current_state, valid_actions, 0, 10)

            shop_index = action // 64
            row = (action % 64) // 8
            col = action % 8

            success = current_state.place_shape(shop_index, row, col)
            if not success:
                break
            moves += 1

        return current_state.score

    def rollout_policy(self, state, valid_actions, depth=0, max_depth=10):
        if depth >= max_depth:
            return state.score

        remaining_shapes = 0
        for shape in state.shop:
            if np.any(shape):
                remaining_shapes += 1

        if remaining_shapes > 1 or len(valid_actions) > 5:
            return self.simple_rollout_policy(state, valid_actions)

        best_action = None
        best_score = -1

        for action in valid_actions:
            score = self.evaluate_action_with_uncertainty(state, action, depth + 1, max_depth)
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

            reward = node.single_rollout()
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
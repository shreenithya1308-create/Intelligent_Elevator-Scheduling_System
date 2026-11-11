import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent with valid-action masking."""

    def __init__(self, state_shape=(10, 3, 3, 3, 3, 3, 3, 3, 3), n_actions=3,
                 learning_rate=0.10, discount_factor=0.95,
                 epsilon=1.0, epsilon_min=0.04, epsilon_decay=0.9985,
                 seed=None):
        self.q_table = np.zeros(state_shape + (n_actions,), dtype=np.float32)
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _valid_values(q_values, valid_actions):
        return np.asarray([q_values[a] for a in valid_actions], dtype=np.float32)

    def choose_action(self, state, valid_actions=None, training=True):
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
        if not valid_actions:
            return 2
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))
        values = self._valid_values(self.q_table[state], valid_actions)
        best = np.flatnonzero(values == values.max())
        return int(valid_actions[int(self.rng.choice(best))])

    def learn(self, state, action, reward, next_state, done, next_valid_actions=None):
        old_value = self.q_table[state + (action,)]
        if done:
            target = reward
        else:
            if next_valid_actions is None:
                next_valid_actions = list(range(self.n_actions))
            next_values = self._valid_values(self.q_table[next_state], next_valid_actions)
            target = reward + self.gamma * float(next_values.max())
        self.q_table[state + (action,)] = old_value + self.alpha * (target - old_value)

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="q_table.npy"):
        np.save(path, self.q_table)

    def load(self, path="q_table.npy"):
        loaded = np.load(path)
        if loaded.shape != self.q_table.shape:
            raise ValueError(
                f"Q-table shape mismatch: file {loaded.shape}, expected {self.q_table.shape}. "
                "Retrain the agent with this project version."
            )
        self.q_table = loaded

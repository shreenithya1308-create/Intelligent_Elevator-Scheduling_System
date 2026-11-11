import csv
import numpy as np
import matplotlib.pyplot as plt

from environment import ElevatorEnv
from agent import QLearningAgent

EPISODES = 4000
MAX_STEPS = 400
SEED = 42
WINDOW = 100


def moving_average(values, window=WINDOW):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main():
    env = ElevatorEnv(max_steps=MAX_STEPS, seed=SEED)
    agent = QLearningAgent(seed=SEED)
    history = {"reward": [], "served": [], "average_wait": [], "movement": [],
               "waiting_at_end": [], "invalid_actions": []}

    for episode in range(EPISODES):
        state = env.reset(seed=SEED + episode)
        total_reward = 0.0
        for _ in range(MAX_STEPS):
            valid_actions = env.valid_actions()
            action = agent.choose_action(state, valid_actions, training=True)
            next_state, reward, done, _ = env.step(action)
            next_valid = env.valid_actions()
            agent.learn(state, action, reward, next_state, done, next_valid)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_exploration()
        m = env.metrics()
        history["reward"].append(total_reward)
        history["served"].append(m["served"])
        history["average_wait"].append(m["average_wait"])
        history["movement"].append(m["movement"])
        history["waiting_at_end"].append(m["total_waiting_people"])
        history["invalid_actions"].append(m["invalid_actions"])

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1:4d}/{EPISODES} | "
                f"Reward: {np.mean(history['reward'][-100:]):8.2f} | "
                f"Served: {np.mean(history['served'][-100:]):5.2f} | "
                f"Wait: {np.mean(history['average_wait'][-100:]):5.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    agent.save("q_table.npy")
    with open("training_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", *history.keys()])
        for i in range(EPISODES):
            writer.writerow([i + 1, *[history[k][i] for k in history]])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plots = [
        ("reward", "Episode reward", "Reward"),
        ("served", "Completed passenger journeys", "Passengers"),
        ("average_wait", "Average wait until boarding", "Time steps"),
        ("movement", "Elevator floor movements", "Floors"),
        ("waiting_at_end", "Passengers left waiting at episode end", "Passengers"),
        ("invalid_actions", "Invalid actions (should be zero)", "Count"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, plots):
        values = np.asarray(history[key], dtype=float)
        ax.plot(values, alpha=0.18, linewidth=0.7, label="Episode")
        avg = moving_average(values)
        offset = WINDOW - 1 if len(values) >= WINDOW else 0
        ax.plot(np.arange(len(avg)) + offset, avg, linewidth=2.0,
                label=f"{WINDOW}-episode moving average")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.suptitle("Q-Learning Training Performance — Intelligent Elevator", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("\nTraining complete. Saved q_table.npy, training_metrics.csv and learning_curves.png")


if __name__ == "__main__":
    main()

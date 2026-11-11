import csv
import numpy as np
import matplotlib.pyplot as plt

from environment import ElevatorEnv
from agent import QLearningAgent
from baseline import baseline_action

EPISODES = 100
MAX_STEPS = 400
BASE_SEED = 9000


def run_episode(policy_name, q_table=None, seed=0):
    env = ElevatorEnv(max_steps=MAX_STEPS, seed=seed)
    state = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(MAX_STEPS):
        if policy_name == "RL":
            valid = env.valid_actions()
            values = np.asarray([q_table[state][a] for a in valid])
            action = valid[int(np.argmax(values))]
        else:
            action = baseline_action(env)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    result = env.metrics()
    result["reward"] = total_reward
    return result


def main():
    agent = QLearningAgent()
    agent.load("q_table.npy")
    results = {"Q-Learning": [], "Baseline": []}
    for i in range(EPISODES):
        seed = BASE_SEED + i
        results["Q-Learning"].append(run_episode("RL", agent.q_table, seed))
        results["Baseline"].append(run_episode("baseline", seed=seed))

    metrics = [
        ("served", "Completed journeys", "higher"),
        ("average_wait", "Average wait until boarding", "lower"),
        ("movement", "Floor movements", "context"),
        ("total_waiting_people", "People waiting at end", "lower"),
        ("waiting_time_at_end", "Total age of passengers still waiting", "lower"),
        ("reward", "Cumulative reward", "higher"),
    ]
    means = {}
    for policy, rows in results.items():
        means[policy] = {key: float(np.mean([r.get(key, 0) for r in rows])) for key, _, _ in metrics}

    print(f"\nMEAN RESULTS OVER {EPISODES} MATCHED-SEED EPISODES")
    print("-" * 76)
    print(f"{'Metric':<39}{'Q-Learning':>17}{'Baseline':>17}")
    print("-" * 76)
    for key, label, _ in metrics:
        print(f"{label:<39}{means['Q-Learning'][key]:>17.2f}{means['Baseline'][key]:>17.2f}")

    with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "q_learning", "baseline", "preferred_direction"])
        for key, label, preference in metrics:
            writer.writerow([label, means["Q-Learning"][key], means["Baseline"][key], preference])

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (key, label, _) in zip(axes.flat, metrics):
        vals = [means["Q-Learning"][key], means["Baseline"][key]]
        bars = ax.bar(["Q-Learning", "Baseline"], vals)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.1f}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Matched-seed Evaluation: Learned Policy vs Heuristic Baseline", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("comparison_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved evaluation_results.csv and comparison_dashboard.png")
    print("Note: compare all metrics; a higher reward alone does not establish better real-world service.")


if __name__ == "__main__":
    main()

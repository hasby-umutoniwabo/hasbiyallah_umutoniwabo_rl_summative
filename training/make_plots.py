import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.common import ROOT

ALGOS = ["dqn", "reinforce", "a2c", "ppo"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
COLOR = {"dqn": "#0072B2", "reinforce": "#D55E00", "a2c": "#009E73", "ppo": "#CC79A7"}
TRACKS = ["Health", "Business/Finance", "Education"]

OUT = os.path.join(ROOT, "assets", "plots")
TABLES = os.path.join(ROOT, "assets", "tables")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def read_log(algo, name):
    return pd.read_csv(os.path.join(ROOT, "logs", algo, f"{name}.csv"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("saved", name)


def main():
    summary = pd.read_csv(os.path.join(ROOT, "logs", "summary.csv"))
    algos = [a for a in ALGOS if a in set(summary["algo"])]
    best = {a: summary[summary.algo == a].sort_values(["test_success", "test_return"], ascending=False).iloc[0]
            for a in algos}

    # 1 training reward of the best run for each algorithm
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in algos:
        log = read_log(a, best[a]["name"])
        ax.plot(log["frames"], log["train_return"], color=COLOR[a], lw=1.8,
                label=f"{LABEL[a]} ({best[a]['name']})")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward (mean of last 20)")
    ax.set_title("Cumulative reward of the best run per algorithm")
    ax.legend(frameon=False)
    save(fig, "01_cumulative_reward_best.png")

    # 2 all runs for each algorithm to show the spread of the sweep
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True)
    for ax, a in zip(axes.flat, algos):
        for name in summary[summary.algo == a]["name"]:
            log = read_log(a, name)
            ax.plot(log["frames"], log["train_return"], color=COLOR[a], lw=1, alpha=0.5)
        ax.set_title(f"{LABEL[a]}, all runs")
    for ax in axes[1]:
        ax.set_xlabel("Environment steps")
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode reward")
    save(fig, "02_cumulative_reward_all_runs.png")

    # 3 DQN loss
    if "dqn" in algos:
        fig, ax = plt.subplots(figsize=(7, 4))
        runs = summary[summary.algo == "dqn"]["name"].tolist()
        for name in runs:
            log = read_log("dqn", name).dropna(subset=["loss"])
            is_best = name == best["dqn"]["name"]
            ax.plot(log["frames"], log["loss"], color=COLOR["dqn"] if is_best else "grey",
                    lw=2 if is_best else 0.8, alpha=1 if is_best else 0.4,
                    label=f"best ({name})" if is_best else None)
        ax.set_yscale("log")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("TD loss (log scale)")
        ax.set_title("DQN loss, best run highlighted")
        ax.legend(frameon=False)
        save(fig, "03_dqn_loss.png")

    # 4 policy entropy for the policy gradient methods
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in ["reinforce", "a2c", "ppo"]:
        if a not in algos:
            continue
        # use the best run that logged entropy
        cand = summary[summary.algo == a].sort_values(["test_success", "test_return"], ascending=False)
        for name in cand["name"]:
            log = read_log(a, name).dropna(subset=["entropy"])
            if len(log):
                ax.plot(log["frames"], log["entropy"], color=COLOR[a], lw=1.8, label=f"{LABEL[a]} ({name})")
                break
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Policy entropy")
    ax.set_title("Policy entropy during training")
    ax.legend(frameon=False)
    save(fig, "04_policy_entropy.png")

    # 5 convergence, success rate on evaluation learners
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in algos:
        log = read_log(a, best[a]["name"]).dropna(subset=["eval_success_rate"])
        ax.plot(log["frames"], log["eval_success_rate"], color=COLOR[a], lw=1.8, label=LABEL[a])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Share of learners reaching mastery")
    ax.set_title("Convergence of the best run per algorithm")
    ax.legend(frameon=False)
    save(fig, "05_convergence_success.png")

    # 6 generalisation to unseen learners, split by track
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.2
    x = np.arange(3)
    for i, a in enumerate(algos):
        vals = [best[a][f"success_track{t}"] for t in range(3)]
        ax.bar(x + i * width, vals, width, color=COLOR[a], label=LABEL[a])
    ax.set_xticks(x + width * (len(algos) - 1) / 2)
    ax.set_xticklabels(TRACKS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mastery rate on unseen learners")
    ax.set_title("Generalisation by learner track (best run per algorithm)")
    ax.legend(frameon=False, ncol=4, loc="upper center")
    save(fig, "06_generalisation_by_track.png")

    # 7 every run, test success on unseen learners
    fig, axes = plt.subplots(1, len(algos), figsize=(3.2 * len(algos), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, algos):
        g = summary[summary.algo == a]
        ax.bar([n.split("_")[1] for n in g["name"]], g["test_success"], color=COLOR[a])
        ax.set_title(LABEL[a])
        ax.set_xlabel("Run number")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Mastery rate on 100 unseen learners")
    save(fig, "07_all_runs_test_success.png")

    # hyperparameter tables for the report, one markdown table per algorithm
    for a in algos:
        g = summary[summary.algo == a][["name", "settings", "test_return", "test_success", "test_dropout",
                                        "test_sessions"]].copy()
        g["test_return"] = g["test_return"].round(2)
        g["test_sessions"] = g["test_sessions"].round(1)
        lines = ["| Run | Settings changed or used | Test return | Mastery | Dropout | Sessions |",
                 "|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r['name']} | {r['settings']} | {r['test_return']} | {r['test_success']:.2f} | "
                         f"{r['test_dropout']:.2f} | {r['test_sessions']} |")
        with open(os.path.join(TABLES, f"{a}_table.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
    print("tables saved in assets/tables")


if __name__ == "__main__":
    main()
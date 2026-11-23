import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from training.common import ROOT, evaluate
from training.policy_io import load_policy

ALGOS = ["dqn", "reinforce", "a2c", "ppo"]
SKIP = {"name", "algo", "seed"}


def main():
    rows = []
    for algo in ALGOS:
        folder = "dqn" if algo == "dqn" else "pg"
        log_dir = os.path.join(ROOT, "logs", algo)
        if not os.path.isdir(log_dir):
            continue
        for fname in sorted(os.listdir(log_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(log_dir, fname)) as f:
                cfg = json.load(f)
            name = cfg["name"]
            path = os.path.join(ROOT, "models", folder, f"{name}_best.pt")
            if not os.path.exists(path):
                continue
            act = load_policy(algo, path, cfg["hidden"])
            # test seeds are different from the ones used during training and model selection
            res = evaluate(act, episodes=100, seed0=30_000)
            row = {"algo": algo, "name": name,
                   "settings": "; ".join(f"{k}={v}" for k, v in cfg.items() if k not in SKIP | {"hidden"}),
                   "hidden": cfg["hidden"],
                   "test_return": res["mean_return"], "test_success": res["success_rate"],
                   "test_dropout": res["dropout_rate"], "test_sessions": res["mean_sessions"]}
            for t in range(3):
                r = evaluate(act, episodes=30, seed0=40_000, track=t)
                row[f"success_track{t}"] = r["success_rate"]
            rows.append(row)
            print(f"{name}: return {res['mean_return']:.2f}, success {res['success_rate']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "logs", "summary.csv"), index=False)

    # pick the best run per algorithm, then the best overall
    best = {}
    for algo, g in df.groupby("algo"):
        top = g.sort_values(["test_success", "test_return"], ascending=False).iloc[0]
        folder = "dqn" if algo == "dqn" else "pg"
        best[algo] = {"algo": algo, "name": top["name"], "hidden": int(top["hidden"]),
                      "path": f"models/{folder}/{top['name']}_best.pt",
                      "test_success": float(top["test_success"]),
                      "test_return": float(top["test_return"])}
    overall = max(best.values(), key=lambda b: (b["test_success"], b["test_return"]))
    best["overall"] = overall
    with open(os.path.join(ROOT, "models", "best_models.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("\nbest overall:", overall["name"], f"(success {overall['test_success']:.2f})")


if __name__ == "__main__":
    main()
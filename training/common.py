import csv
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from torchrl.envs import GymWrapper, RewardSum, TransformedEnv

from environment.custom_env import AdaptiveTutorEnv


def make_env(seed=0):
    # integer actions, and RewardSum gives us the episode return
    env = TransformedEnv(
        GymWrapper(AdaptiveTutorEnv(), categorical_action_encoding=True),
        RewardSum(),
    )
    env.set_seed(seed)
    return env


def set_seed(seed):
    # tiny networks run faster on one thread
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(act_fn, episodes=30, seed0=10_000, track=None):
    # greedy rollouts on seeds the agent never trained on
    env = AdaptiveTutorEnv()
    returns, sessions, outcomes = [], [], []
    for i in range(episodes):
        options = {"track": track} if track is not None else None
        obs, _ = env.reset(seed=seed0 + i, options=options)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act_fn(obs))
            total += r
            done = term or trunc
        returns.append(total)
        sessions.append(info["sessions"])
        outcomes.append(info["outcome"])
    return {
        "mean_return": float(np.mean(returns)),
        "success_rate": outcomes.count("mastered") / episodes,
        "dropout_rate": outcomes.count("dropped_out") / episodes,
        "mean_sessions": float(np.mean(sessions)),
    }


# eval columns stay empty on rows without an evaluation
EMPTY_EVAL = {
    "eval_mean_return": float("nan"),
    "eval_success_rate": float("nan"),
    "eval_dropout_rate": float("nan"),
    "eval_mean_sessions": float("nan"),
}


class Logger:
    def __init__(self, algo, name, cfg):
        self.dir = os.path.join(ROOT, "logs", algo)
        os.makedirs(self.dir, exist_ok=True)
        self.csv_path = os.path.join(self.dir, f"{name}.csv")
        with open(os.path.join(self.dir, f"{name}.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        self.file = open(self.csv_path, "w", newline="")
        self.writer = None

    def log(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row.keys()))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def model_path(folder, name):
    path = os.path.join(ROOT, "models", folder)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, name)
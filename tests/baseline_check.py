import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gymnasium.utils.env_checker import check_env
from environment.custom_env import AdaptiveTutorEnv, N_ACTIONS

env = AdaptiveTutorEnv()
check_env(env)
print("Gymnasium check_env: PASSED")


def run(policy, n=300):
    out = {"mastered": 0, "dropped_out": 0, "timeout": 0}
    rewards, lengths = [], []
    for i in range(n):
        obs, _ = env.reset(seed=i)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            total += r
            done = term or trunc
        out[info["outcome"]] += 1
        rewards.append(total)
        lengths.append(info["sessions"])
    return out, round(float(np.mean(rewards)), 2), round(float(np.mean(lengths)), 1)


rng = np.random.default_rng(0)


def heuristic(o):
    if o[5] > 0.6:          # tired -> break
        return 12
    s = int(np.argmin(o[:4]))  # weakest skill
    d = 0 if o[s] < 0.3 else 1 if o[s] < 0.6 else 2
    return s * 3 + d


print("random   :", run(lambda o: int(rng.integers(N_ACTIONS))))
print("heuristic:", run(heuristic))
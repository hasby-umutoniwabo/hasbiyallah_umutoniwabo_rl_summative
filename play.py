import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pygame

from environment.custom_env import ACTION_NAMES, N_ACTIONS, AdaptiveTutorEnv
from training.policy_io import load_policy


def load_agent(algo):
    with open(os.path.join(ROOT, "models", "best_models.json")) as f:
        info = json.load(f)
    entry = info[algo] if algo else info["overall"]
    act = load_policy(entry["algo"], os.path.join(ROOT, entry["path"]), entry["hidden"])
    return act, entry


def main():
    p = argparse.ArgumentParser(description="Watch the trained tutor agent teach simulated learners")
    p.add_argument("--algo", choices=["dqn", "reinforce", "a2c", "ppo"], help="default is the best model overall")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--track", type=int, choices=[0, 1, 2], help="0 health, 1 business, 2 education")
    p.add_argument("--random", action="store_true", help="random agent for comparison")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--fps", type=int, default=3, help="sessions per second in the window")
    args = p.parse_args()

    if args.random:
        rng = np.random.default_rng(0)
        act, label = (lambda obs: int(rng.integers(N_ACTIONS))), "random agent"
    else:
        act, entry = load_agent(args.algo)
        label = f"{entry['algo'].upper()} ({entry['name']})"

    env = AdaptiveTutorEnv(render_mode=None if args.no_render else "human")
    env.metadata = {**env.metadata, "render_fps": args.fps}
    print(f"Agent: {label}")
    results = []
    for ep in range(args.episodes):
        options = {"track": args.track} if args.track is not None else None
        obs, info = env.reset(seed=args.seed + ep, options=options)
        print(f"\nEpisode {ep + 1} | learner track: {info['track']}")
        done, total = False, 0.0
        while not done:
            action = act(obs)
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            done = term or trunc
            print(f"  session {info['sessions']:2d} | {ACTION_NAMES[action]:<24} | reward {reward:+6.2f} | "
                  f"motivation {info['motivation']:.2f} | fatigue {info['fatigue']:.2f} | "
                  f"readiness {info['weighted_mastery']:.2f}")
        results.append((info["outcome"], total, info["sessions"]))
        print(f"  result: {info['outcome']} after {info['sessions']} sessions, total reward {total:.2f}")
        if not args.no_render:
            pygame.time.wait(1500)

    mastered = sum(r[0] == "mastered" for r in results)
    print(f"\nSummary: {mastered}/{len(results)} learners reached workplace readiness, "
          f"average reward {np.mean([r[1] for r in results]):.2f}")
    env.close()


if __name__ == "__main__":
    main()
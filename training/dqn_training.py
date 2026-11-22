import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch import nn
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate

from environment.custom_env import N_ACTIONS, OBS_DIM
from training.common import EMPTY_EVAL, Logger, evaluate, make_env, model_path, set_seed

BASE = dict(
    lr=1e-3,
    gamma=0.99,
    buffer=50_000,
    batch=64,
    eps_steps=20_000,
    target_every=500,
    hidden=128,
    double=False,
)

# every run changes one thing from the baseline (run 1)
CHANGES = [
    {},
    {"lr": 3e-4},
    {"lr": 3e-3},
    {"gamma": 0.95},
    {"gamma": 0.999},
    {"buffer": 10_000},
    {"batch": 256},
    {"eps_steps": 5_000},
    {"eps_steps": 50_000},
    {"target_every": 100},
    {"target_every": 2_000},
    {"double": True, "hidden": 256},
]
CONFIGS = [{"name": f"dqn_{i + 1:02d}", **BASE, **c} for i, c in enumerate(CHANGES)]


def run(cfg, total_frames, seed):
    set_seed(seed)
    env = make_env(seed)

    qnet = MLP(
        in_features=OBS_DIM,
        out_features=N_ACTIONS,
        num_cells=[cfg["hidden"], cfg["hidden"]],
        activation_class=nn.ReLU,
    )
    actor = QValueActor(qnet, in_keys=["observation"], spec=env.action_spec, action_space="categorical")
    egreedy = EGreedyModule(env.action_spec, annealing_num_steps=cfg["eps_steps"], eps_init=1.0, eps_end=0.05)
    explore = TensorDictSequential(actor, egreedy)

    frames_per_batch = 200
    collector = Collector(env, explore, frames_per_batch=frames_per_batch, total_frames=total_frames)
    buffer = ReplayBuffer(storage=LazyTensorStorage(cfg["buffer"]))

    loss_fn = DQNLoss(actor, action_space="categorical", delay_value=True, double_dqn=cfg["double"])
    loss_fn.make_value_estimator(gamma=cfg["gamma"])
    updater = HardUpdate(loss_fn, value_network_update_interval=cfg["target_every"])
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    def act(obs):
        with torch.no_grad():
            return int(qnet(torch.as_tensor(obs)).argmax())

    logger = Logger("dqn", cfg["name"], {**cfg, "seed": seed, "total_frames": total_frames})
    best = -1e9
    frames, recent, start = 0, [], time.time()

    for i, batch in enumerate(collector):
        frames += batch.numel()
        buffer.extend(batch.reshape(-1))

        done = batch["next", "done"].squeeze(-1)
        recent += batch["next", "episode_reward"][done].squeeze(-1).tolist()
        recent = recent[-20:]

        losses = []
        if frames >= 1000:
            for _ in range(50):
                sample = buffer.sample(cfg["batch"])
                loss = loss_fn(sample)["loss"]
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_fn.parameters(), 10.0)
                optim.step()
                updater.step()
                losses.append(loss.item())
        egreedy.step(batch.numel())

        row = {
            "frames": frames,
            "train_return": float(np.mean(recent)) if recent else float("nan"),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "epsilon": float(egreedy.eps),
            **EMPTY_EVAL,
        }
        if (i + 1) % 5 == 0:
            ev = evaluate(act, episodes=20)
            row.update({f"eval_{k}": v for k, v in ev.items()})
            if ev["mean_return"] > best:
                best = ev["mean_return"]
                torch.save(qnet.state_dict(), model_path("dqn", f"{cfg['name']}_best.pt"))
        logger.log(row)

    torch.save(qnet.state_dict(), model_path("dqn", f"{cfg['name']}_final.pt"))
    final = evaluate(act, episodes=50, seed0=20_000)
    logger.close()
    collector.shutdown()
    print(
        f"{cfg['name']} done in {time.time() - start:.0f}s | "
        f"return {final['mean_return']:.2f} | success {final['success_rate']:.2f}"
    )
    return {"name": cfg["name"], **final}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=int, nargs="*", help="run numbers, for example 1 2 3")
    p.add_argument("--all", action="store_true")
    p.add_argument("--frames", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    chosen = CONFIGS if args.all or not args.run else [CONFIGS[n - 1] for n in args.run]
    for cfg in chosen:
        run(cfg, args.frames, args.seed)
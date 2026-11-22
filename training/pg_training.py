import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.objectives.value.functional import reward2go

from environment.custom_env import N_ACTIONS, OBS_DIM
from training.common import EMPTY_EVAL, Logger, evaluate, make_env, model_path, set_seed

# baseline settings for each algorithm, every run changes one thing
BASES = {
    "reinforce": dict(lr=1e-3, gamma=0.99, entropy=0.01, episodes=8, hidden=128, baseline=True),
    "a2c": dict(lr=1e-3, gamma=0.99, lmbda=0.95, entropy=0.01, critic=0.5, rollout=400, hidden=128),
    "ppo": dict(lr=3e-4, gamma=0.99, lmbda=0.95, entropy=0.01, critic=0.5, clip=0.2,
                rollout=1000, epochs=8, minibatch=100, hidden=128),
}

CHANGES = {
    "reinforce": [
        {},
        {"lr": 3e-4},
        {"lr": 3e-3},
        {"gamma": 0.95},
        {"gamma": 0.999},
        {"entropy": 0.0},
        {"entropy": 0.05},
        {"episodes": 2},
        {"episodes": 20},
        {"baseline": False},
    ],
    "a2c": [
        {},
        {"lr": 3e-4},
        {"lr": 3e-3},
        {"gamma": 0.95},
        {"gamma": 0.999},
        {"entropy": 0.0},
        {"entropy": 0.05},
        {"rollout": 100},
        {"rollout": 1600},
        {"critic": 1.0, "lmbda": 0.9},
    ],
    "ppo": [
        {},
        {"lr": 1e-4},
        {"lr": 1e-3},
        {"gamma": 0.95},
        {"gamma": 0.999},
        {"entropy": 0.0},
        {"entropy": 0.05},
        {"clip": 0.1},
        {"clip": 0.3},
        {"rollout": 400, "epochs": 4},
    ],
}


def get_configs(algo):
    return [{"name": f"{algo}_{i + 1:02d}", "algo": algo, **BASES[algo], **c}
            for i, c in enumerate(CHANGES[algo])]


def build_networks(env, hidden):
    policy_net = MLP(in_features=OBS_DIM, out_features=N_ACTIONS,
                     num_cells=[hidden, hidden], activation_class=nn.Tanh)
    actor = ProbabilisticActor(
        TensorDictModule(policy_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"],
        spec=env.action_spec,
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )
    value_net = MLP(in_features=OBS_DIM, out_features=1, num_cells=[hidden, hidden], activation_class=nn.Tanh)
    critic = ValueOperator(value_net, in_keys=["observation"])
    return policy_net, actor, critic


def run(cfg, total_frames, seed):
    algo = cfg["algo"]
    set_seed(seed)
    env = make_env(seed)
    policy_net, actor, critic = build_networks(env, cfg["hidden"])

    def act(obs):
        with torch.no_grad():
            return int(policy_net(torch.as_tensor(obs)).argmax())

    logger = Logger(algo, cfg["name"], {**cfg, "seed": seed, "total_frames": total_frames})
    best, frames, recent, start, n_updates = -1e9, 0, [], time.time(), 0

    if algo == "reinforce":
        params = list(actor.parameters()) + (list(critic.parameters()) if cfg["baseline"] else [])
        optim = torch.optim.Adam(params, lr=cfg["lr"])
    else:
        advantage = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic, average_gae=True)
        if algo == "ppo":
            loss_fn = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip"], entropy_coeff=cfg["entropy"],
                                  critic_coeff=cfg["critic"], normalize_advantage=True)
        else:
            loss_fn = A2CLoss(actor, critic, entropy_coeff=cfg["entropy"], critic_coeff=cfg["critic"])
        optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
        collector = Collector(env, actor, frames_per_batch=cfg["rollout"], total_frames=total_frames,
                              auto_register_policy_transforms=False)

    def log_and_eval(row):
        nonlocal best
        row = {**row, **EMPTY_EVAL}
        if n_updates % 5 == 0:
            ev = evaluate(act, episodes=20)
            row.update({f"eval_{k}": v for k, v in ev.items()})
            if ev["mean_return"] > best:
                best = ev["mean_return"]
                torch.save(policy_net.state_dict(), model_path("pg", f"{cfg['name']}_best.pt"))
        logger.log(row)

    if algo == "reinforce":
        while frames < total_frames:
            # one update uses several full episodes
            rewards, logps, ents, values = [], [], [], []
            for _ in range(cfg["episodes"]):
                td = env.rollout(max_steps=200, policy=actor)
                frames += td.shape[0]
                recent.append(float(td["next", "episode_reward"][-1]))
                g = reward2go(td["next", "reward"], td["next", "done"], gamma=cfg["gamma"])
                dist = torch.distributions.Categorical(logits=policy_net(td["observation"]))
                logps.append(dist.log_prob(td["action"]))
                ents.append(dist.entropy())
                rewards.append(g.squeeze(-1))
                if cfg["baseline"]:
                    values.append(critic(td.select("observation"))["state_value"].squeeze(-1))
            recent = recent[-20:]
            g = torch.cat(rewards)
            logp = torch.cat(logps)
            ent = torch.cat(ents).mean()
            if cfg["baseline"]:
                v = torch.cat(values)
                adv = (g - v.detach())
                critic_loss = nn.functional.mse_loss(v, g)
            else:
                adv, critic_loss = g, 0.0
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            loss = -(logp * adv).mean() - cfg["entropy"] * ent + 0.5 * critic_loss
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            n_updates += 1
            log_and_eval({"frames": frames, "train_return": float(np.mean(recent)),
                          "loss": float(loss.item()), "entropy": float(ent.item())})
    else:
        for batch in collector:
            frames += batch.numel()
            done = batch["next", "done"].squeeze(-1)
            recent += batch["next", "episode_reward"][done].squeeze(-1).tolist()
            recent = recent[-20:]
            with torch.no_grad():
                advantage(batch)
            flat = batch.reshape(-1)
            losses, ents = [], []
            epochs = cfg["epochs"] if algo == "ppo" else 1
            mb = cfg["minibatch"] if algo == "ppo" else len(flat)
            for _ in range(epochs):
                perm = torch.randperm(len(flat))
                for k in range(0, len(flat), mb):
                    out = loss_fn(flat[perm[k:k + mb]])
                    loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(loss_fn.parameters(), 0.5)
                    optim.step()
                    losses.append(loss.item())
                    ents.append(float(out["entropy"]))
            n_updates += 1
            log_and_eval({"frames": frames,
                          "train_return": float(np.mean(recent)) if recent else float("nan"),
                          "loss": float(np.mean(losses)), "entropy": float(np.mean(ents))})
        collector.shutdown()

    torch.save(policy_net.state_dict(), model_path("pg", f"{cfg['name']}_final.pt"))
    final = evaluate(act, episodes=50, seed0=20_000)
    logger.close()
    print(f"{cfg['name']} done in {time.time() - start:.0f}s | "
          f"return {final['mean_return']:.2f} | success {final['success_rate']:.2f}")
    return {"name": cfg["name"], **final}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["reinforce", "a2c", "ppo"], required=True)
    p.add_argument("--run", type=int, nargs="*", help="run numbers, for example 1 2 3")
    p.add_argument("--all", action="store_true")
    p.add_argument("--frames", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    configs = get_configs(args.algo)
    chosen = configs if args.all or not args.run else [configs[n - 1] for n in args.run]
    for cfg in chosen:
        run(cfg, args.frames, args.seed)
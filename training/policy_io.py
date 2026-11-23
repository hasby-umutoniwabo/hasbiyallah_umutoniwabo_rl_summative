import os
import sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from torchrl.modules import MLP

from environment.custom_env import N_ACTIONS, OBS_DIM


def load_policy(algo, path, hidden):
    # DQN used ReLU, the policy gradient networks used Tanh
    act_class = nn.ReLU if algo == "dqn" else nn.Tanh
    net = MLP(in_features=OBS_DIM, out_features=N_ACTIONS,
              num_cells=[hidden, hidden], activation_class=act_class)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()

    def act(obs):
        # greedy action: best Q value for DQN, most likely action for the others
        with torch.no_grad():
            return int(net(torch.as_tensor(obs)).argmax())

    return act
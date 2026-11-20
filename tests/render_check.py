import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
from environment.custom_env import AdaptiveTutorEnv


def heuristic(o):
    if o[5] > 0.6:
        return 12
    s = int(np.argmin(o[:4]))
    d = 0 if o[s] < 0.3 else 1 if o[s] < 0.6 else 2
    return s * 3 + d


env = AdaptiveTutorEnv(render_mode="human")
obs, _ = env.reset(seed=3)
done = False
step = 0
while not done:
    obs, r, term, trunc, info = env.step(heuristic(obs))
    done = term or trunc
    step += 1
    if step == 15:
        # screenshot for the report
        os.makedirs("assets", exist_ok=True)
        pygame.image.save(env._renderer.screen, os.path.join("assets", "env_screenshot.png"))
pygame.time.wait(2500)
env.close()
print("episode ended:", info["outcome"], "after", info["sessions"], "sessions")
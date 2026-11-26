import os
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

from environment.interactive import InteractiveApp
from environment.rendering import action_rects, header_pills
from play import load_agent

# simulates mouse clicks and key presses without opening a real window
act, entry = load_agent("ppo")
app = InteractiveApp(act, f"{entry['algo'].upper()} ({entry['name']})")
click = lambda rect: app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=rect.center, button=1))
key = lambda k: app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k))

click(action_rects()[0])  # Easy file management
click(action_rects()[7])  # Word... Excel medium
key(pygame.K_b)
key(pygame.K_SPACE)
assert app.env.t == 4, app.env.t
click(header_pills()["auto"])
assert app.auto
for _ in range(2000):
    app.update(0.1)
    if app.done:
        break
app.draw()
out = os.path.join(tempfile.gettempdir(), "interactive_check.png")
pygame.image.save(app.renderer.screen, out)
assert app.done and app.results
print("interactive check passed:", app.results[-1])
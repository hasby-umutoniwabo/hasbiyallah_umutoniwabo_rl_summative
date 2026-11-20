import numpy as np
import pygame

from environment.custom_env import (
    ACTION_NAMES,
    DIFFICULTIES,
    SKILLS,
    TRACK_WEIGHTS,
    TRACKS,
)

W, H = 1000, 620
BG = (24, 28, 38)
PANEL = (36, 42, 56)
TEXT = (232, 236, 244)
MUTED = (140, 150, 170)
GREEN = (80, 200, 120)
RED = (230, 90, 90)
BLUE = (90, 150, 240)
AMBER = (240, 190, 70)
DIFF_COLORS = [(70, 160, 110), (200, 160, 60), (200, 90, 80)]
SKILL_COLORS = [(90, 150, 240), (150, 110, 230), (80, 200, 160), (240, 140, 90)]


class TutorRenderer:
    def __init__(self, mode, fps):
        self.mode = mode
        self.fps = fps
        pygame.init()
        pygame.font.init()
        # window for demos, plain surface for screenshots
        if mode == "human":
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Adaptive Digital Skills Tutor")
        else:
            self.screen = pygame.Surface((W, H))
        self.clock = pygame.time.Clock()
        self.big = pygame.font.SysFont("arial", 26, bold=True)
        self.mid = pygame.font.SysFont("arial", 18, bold=True)
        self.small = pygame.font.SysFont("arial", 15)

    def _text(self, s, font, color, pos):
        self.screen.blit(font.render(s, True, color), pos)

    def _bar(self, x, y, w, h, value, color, goal=None):
        pygame.draw.rect(self.screen, (55, 62, 80), (x, y, w, h), border_radius=5)
        fill = int(w * max(0.0, min(1.0, value)))
        if fill > 0:
            pygame.draw.rect(self.screen, color, (x, y, fill, h), border_radius=5)
        if goal is not None:
            gx = x + int(w * goal)
            pygame.draw.line(self.screen, TEXT, (gx, y - 3), (gx, y + h + 3), 2)

    def _panel(self, rect, title):
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
        self._text(title, self.mid, TEXT, (rect[0] + 14, rect[1] + 10))

    def draw(self, env):
        # let the window close properly
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        self.screen.fill(BG)
        self._text("Adaptive Digital Skills Tutor", self.big, TEXT, (20, 14))
        self._text("RL agent picks the next exercise for a simulated learner", self.small, MUTED, (20, 48))

        # learner panel
        self._panel((20, 80, 300, 400), "Learner")
        x = 36
        self._text(f"Track: {TRACKS[env.track]}", self.small, TEXT, (x, 120))
        self._text(f"Session {env.t} of {env.max_sessions}", self.small, MUTED, (x, 142))

        self._text("Motivation", self.small, MUTED, (x, 180))
        mcol = GREEN if env.motivation > 0.4 else AMBER if env.motivation > 0.2 else RED
        self._bar(x, 202, 268, 16, env.motivation, mcol)

        self._text("Fatigue", self.small, MUTED, (x, 238))
        fcol = GREEN if env.fatigue < 0.4 else AMBER if env.fatigue < 0.7 else RED
        self._bar(x, 260, 268, 16, env.fatigue, fcol)

        mastery = env._weighted_mastery()
        self._text(f"Workplace readiness  {mastery:.2f} / {env.mastery_goal:.2f}", self.small, MUTED, (x, 296))
        self._bar(x, 318, 268, 16, mastery, BLUE, goal=env.mastery_goal)

        self._text(f"Last reward: {env.last_reward:+.2f}", self.small, TEXT, (x, 366))
        self._text(f"Episode reward: {env.total_reward:+.2f}", self.small, TEXT, (x, 390))
        self._text("White line marks the goal", self.small, MUTED, (x, 440))

        # skills panel
        self._panel((340, 80, 300, 400), "Skills")
        weights = TRACK_WEIGHTS[env.track]
        for i, name in enumerate(SKILLS):
            y = 126 + i * 82
            self._text(name, self.small, TEXT, (356, y))
            self._text(f"weight {weights[i]:.2f}", self.small, MUTED, (544, y))
            self._bar(356, y + 26, 268, 18, env.skills[i], SKILL_COLORS[i], goal=env.min_skill_goal)
            self._text(f"{env.skills[i]:.2f}", self.small, MUTED, (356, y + 48))
        self._text("White line is the minimum per skill", self.small, MUTED, (356, 450))

        # actions panel, last action gets a white border
        self._panel((660, 80, 320, 400), "Tutor actions")
        for a, name in enumerate(ACTION_NAMES):
            if a < 12:
                col, row = a // 3, a % 3
                bx, by = 676 + col * 74, 148 + row * 40
                color = DIFF_COLORS[row]
                label = DIFFICULTIES[row][0]
                self._text(SKILLS[col].split()[0], self.small, MUTED, (bx + 4, 126))
                size = (68, 34)
            else:
                bx, by = 676, 284 + (a - 12) * 44
                color = (90, 100, 130)
                label = "Break" if a == 12 else "Review weakest skill"
                size = (296, 36)
            rect = (bx, by, *size)
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            if env.last_action == a:
                pygame.draw.rect(self.screen, TEXT, rect, 3, border_radius=6)
            self._text(label, self.small, (15, 18, 24) if a < 12 else TEXT, (bx + 8, by + 8))
        self._text("E easy, M medium, H hard", self.small, MUTED, (676, 390))

        # banner turns green, red or amber when the episode ends
        color = {"mastered": GREEN, "dropped_out": RED, "timeout": AMBER}.get(env.outcome, (60, 70, 95))
        pygame.draw.rect(self.screen, color, (20, 500, 960, 70), border_radius=10)
        banner_text = (15, 18, 24) if env.outcome != "running" else TEXT
        self._text(env.last_event, self.big, banner_text, (36, 520))

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2))

    def close(self):
        pygame.quit()
import math

import numpy as np
import pygame

from environment.custom_env import DIFFICULTIES, SKILLS, TRACK_WEIGHTS, TRACKS

W, H = 1100, 640
BG = (240, 243, 248)
CARD = (255, 255, 255)
LINE = (222, 228, 238)
TRACK_BG = (228, 233, 242)
TEXT = (28, 36, 52)
MUTED = (112, 124, 148)
GREEN = (46, 176, 110)
AMBER = (240, 168, 40)
RED = (230, 82, 82)
BLUE = (60, 120, 230)
SKILL_COLORS = [(66, 133, 244), (140, 95, 230), (32, 178, 140), (245, 140, 70)]
DIFF_COLORS = [(46, 176, 110), (240, 168, 40), (230, 82, 82)]
TRACK_COLORS = [(32, 178, 140), (66, 133, 244), (245, 140, 70)]
SKIN = (255, 222, 178)
SUB = 8  # smooth frames drawn for each session in the window


class TutorRenderer:
    def __init__(self, mode, fps):
        self.mode = mode
        self.fps = fps
        pygame.init()
        pygame.font.init()
        if mode == "human":
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Adaptive Digital Skills Tutor")
        else:
            self.screen = pygame.Surface((W, H))
        self.clock = pygame.time.Clock()
        self.f_title = pygame.font.SysFont("arial", 28, bold=True)
        self.f_head = pygame.font.SysFont("arial", 20, bold=True)
        self.f_huge = pygame.font.SysFont("arial", 38, bold=True)
        self.f_big = pygame.font.SysFont("arial", 26, bold=True)
        self.f_body = pygame.font.SysFont("arial", 18)
        self.f_small = pygame.font.SysFont("arial", 14)
        self.shown = None

    def _text(self, s, font, color, pos, right=False, center=False):
        img = font.render(s, True, color)
        rect = img.get_rect()
        if right:
            rect.topright = pos
        elif center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.screen.blit(img, rect)

    def _card(self, x, y, w, h, title=None):
        pygame.draw.rect(self.screen, CARD, (x, y, w, h), border_radius=16)
        pygame.draw.rect(self.screen, LINE, (x, y, w, h), 1, border_radius=16)
        if title:
            self._text(title, self.f_head, TEXT, (x + 24, y + 20))

    def _bar(self, x, y, w, h, value, color, goal=None):
        pygame.draw.rect(self.screen, TRACK_BG, (x, y, w, h), border_radius=h // 2)
        fill = int(w * max(0.0, min(1.0, value)))
        if fill >= h:
            pygame.draw.rect(self.screen, color, (x, y, fill, h), border_radius=h // 2)
        elif fill > 0:
            pygame.draw.rect(self.screen, color, (x, y, h, h), border_radius=h // 2)
        if goal is not None:
            gx = x + int(w * goal)
            pygame.draw.line(self.screen, TEXT, (gx, y - 4), (gx, y + h + 4), 2)

    def _face(self, cx, cy, motivation, energy):
        pygame.draw.circle(self.screen, SKIN, (cx, cy), 62)
        pygame.draw.circle(self.screen, (232, 190, 140), (cx, cy), 62, 3)
        # eyes: open when rested, flat lines when tired
        for dx in (-22, 22):
            if energy < 0.35:
                pygame.draw.line(self.screen, TEXT, (cx + dx - 7, cy - 8), (cx + dx + 7, cy - 8), 4)
            else:
                pygame.draw.circle(self.screen, TEXT, (cx + dx, cy - 8), 6)
        # mouth follows motivation
        if motivation > 0.6:
            pygame.draw.arc(self.screen, TEXT, (cx - 24, cy + 4, 48, 34), math.pi, 2 * math.pi, 4)
        elif motivation > 0.3:
            pygame.draw.line(self.screen, TEXT, (cx - 16, cy + 26), (cx + 16, cy + 26), 4)
        else:
            pygame.draw.arc(self.screen, TEXT, (cx - 24, cy + 20, 48, 34), 0, math.pi, 4)
        if energy < 0.35:
            self._text("z z", self.f_big, MUTED, (cx + 54, cy - 78))

    def _targets(self, env):
        return {
            "motivation": float(env.motivation),
            "energy": 1.0 - float(env.fatigue),
            "ready": float(env._weighted_mastery()),
            **{f"s{i}": float(env.skills[i]) for i in range(len(SKILLS))},
        }

    def draw(self, env):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        target = self._targets(env)
        if self.mode != "human" or self.shown is None or env.t == 0:
            self.shown = dict(target)
            self._paint(env, self.shown)
            if self.mode == "human":
                pygame.display.flip()
                self.clock.tick(self.fps)
                return None
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2))
        # ease the numbers towards the new values so the bars slide
        start = dict(self.shown)
        for k in range(1, SUB + 1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
            t = k / SUB
            t = t * t * (3 - 2 * t)
            self.shown = {key: start[key] + (target[key] - start[key]) * t for key in target}
            self._paint(env, self.shown)
            pygame.display.flip()
            self.clock.tick(self.fps * SUB)
        return None

    def _paint(self, env, d):
        self.screen.fill(BG)

        # header
        self._text("Adaptive Digital Skills Tutor", self.f_title, TEXT, (40, 24))
        self._text(f"Session {env.t} of {env.max_sessions}", self.f_head, MUTED, (W - 40, 30), right=True)

        top, ch = 92, 378

        # learner
        x, w = 40, 330
        self._card(x, top, w, ch)
        self._face(x + w // 2, top + 96, d["motivation"], d["energy"])
        tcol = TRACK_COLORS[env.track]
        chip = pygame.Rect(0, 0, 190, 30)
        chip.center = (x + w // 2, top + 192)
        pygame.draw.rect(self.screen, tcol, chip, border_radius=15)
        self._text(f"{TRACKS[env.track]} learner", self.f_body, (255, 255, 255), chip.center, center=True)
        mcol = GREEN if d["motivation"] > 0.4 else AMBER if d["motivation"] > 0.2 else RED
        ecol = GREEN if d["energy"] > 0.6 else AMBER if d["energy"] > 0.3 else RED
        inner = w - 48
        self._text("Motivation", self.f_body, MUTED, (x + 24, top + 236))
        self._text(f"{d['motivation'] * 100:.0f}%", self.f_body, TEXT, (x + w - 24, top + 236), right=True)
        self._bar(x + 24, top + 264, inner, 14, d["motivation"], mcol)
        self._text("Energy", self.f_body, MUTED, (x + 24, top + 296))
        self._text(f"{d['energy'] * 100:.0f}%", self.f_body, TEXT, (x + w - 24, top + 296), right=True)
        self._bar(x + 24, top + 324, inner, 14, d["energy"], ecol)

        # skills
        x, w = 392, 316
        self._card(x, top, w, ch, "Skills")
        weights = TRACK_WEIGHTS[env.track]
        for i, name in enumerate(SKILLS):
            y = top + 64 + i * 70
            self._text(name, self.f_body, TEXT, (x + 24, y))
            if weights[i] >= 0.30:
                self._text("key skill", self.f_small, AMBER, (x + 24 + self.f_body.size(name)[0] + 10, y + 3))
            self._text(f"{d[f's{i}']:.2f}", self.f_body, TEXT, (x + w - 24, y), right=True)
            col = GREEN if d[f"s{i}"] >= env.min_skill_goal else SKILL_COLORS[i]
            self._bar(x + 24, y + 30, w - 48, 16, d[f"s{i}"], col, goal=env.min_skill_goal)
        self._text("Black line is the target for each skill", self.f_small, MUTED, (x + 24, top + ch - 34))

        # tutor choice
        x, w = 730, 330
        self._card(x, top, w, ch, "Tutor's choice")
        a = env.last_action
        card = pygame.Rect(x + 24, top + 62, w - 48, 150)
        if a is None:
            pygame.draw.rect(self.screen, TRACK_BG, card, border_radius=14)
            self._text("Waiting to start", self.f_big, MUTED, card.center, center=True)
        elif a < 12:
            pygame.draw.rect(self.screen, DIFF_COLORS[a % 3], card, border_radius=14)
            self._text(SKILLS[a // 3], self.f_huge, (255, 255, 255), (card.centerx, card.centery - 16), center=True)
            self._text(f"{DIFFICULTIES[a % 3]} exercise", self.f_head, (255, 255, 255), (card.centerx, card.centery + 28), center=True)
        else:
            pygame.draw.rect(self.screen, BLUE if a == 12 else (130, 100, 230), card, border_radius=14)
            label = "Take a break" if a == 12 else "Review weakest skill"
            self._text(label, self.f_big, (255, 255, 255), card.center, center=True)
        # what happened
        bub = pygame.Rect(x + 24, top + 240, w - 48, 52)
        if a is None:
            msg, bcol = "Learner has just joined", TRACK_BG
        elif a < 12:
            ok = env.last_success >= 0.99
            msg, bcol = ("Learner passed", GREEN) if ok else ("Too hard, learner failed", RED)
        elif a == 12:
            msg, bcol = "Learner rested", BLUE
        else:
            msg, bcol = "Quick review done", (130, 100, 230)
        pygame.draw.rect(self.screen, bcol, bub, border_radius=26)
        self._text(msg, self.f_body, TEXT if a is None else (255, 255, 255), bub.center, center=True)
        self._text(f"Reward  {env.last_reward:+.2f}", self.f_small, MUTED, (x + 24, top + 312))
        self._text(f"Total  {env.total_reward:+.2f}", self.f_small, MUTED, (x + w - 24, top + 312), right=True)

        # bottom: progress to job ready, or the final result
        by = 490
        if env.outcome == "running":
            self._card(40, by, W - 80, 118)
            self._text("Progress to job-ready", self.f_head, TEXT, (64, by + 22))
            self._text(f"{d['ready'] / env.mastery_goal * 100:.0f}%", self.f_head, TEXT, (W - 64, by + 22), right=True)
            self._bar(64, by + 64, W - 128, 24, d["ready"] / env.mastery_goal, BLUE, goal=1.0)
        else:
            col, msg = {
                "mastered": (GREEN, f"Learner is job-ready after {env.t} sessions!"),
                "dropped_out": (RED, "The learner lost motivation and quit"),
                "timeout": (AMBER, "Ran out of sessions"),
            }[env.outcome]
            pygame.draw.rect(self.screen, col, (40, by, W - 80, 118), border_radius=16)
            self._text(msg, self.f_huge, (255, 255, 255), (W // 2, by + 59), center=True)

    def close(self):
        pygame.quit()
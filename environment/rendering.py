import math

import numpy as np
import pygame

from environment.custom_env import ACTION_NAMES, DIFFICULTIES, SKILLS, TRACK_WEIGHTS, TRACKS

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

# layout of the clickable parts used by the interactive mode
TOP = 92
TUTOR_X, TUTOR_W = 730, 330
TRACK_LABELS = ["Random", "Health", "Business", "Education"]


def action_rects():
    # 12 exercise buttons (4 skills x 3 levels) plus break and review
    rects = {}
    gx, gy = TUTOR_X + 24 + 54, TOP + 84
    for c in range(4):
        for r in range(3):
            rects[c * 3 + r] = pygame.Rect(gx + c * 57, gy + r * 44, 50, 36)
    by = TOP + 84 + 3 * 44 + 14
    rects[12] = pygame.Rect(TUTOR_X + 24, by, 136, 40)
    rects[13] = pygame.Rect(TUTOR_X + 24 + 146, by, 136, 40)
    return rects


def header_pills():
    return {name: pygame.Rect(550 + i * 130, 28, 120, 34) for i, name in enumerate(["hint", "auto", "track", "new"])}


def mix(a, b, k):
    return tuple(int(a[i] * (1 - k) + b[i] * k) for i in range(3))


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

    def _paint(self, env, d, ui=None):
        self.screen.fill(BG)

        # header
        self._text("Adaptive Digital Skills Tutor", self.f_title, TEXT, (40, 24))
        if ui is None:
            self._text(f"Session {env.t} of {env.max_sessions}", self.f_head, MUTED, (W - 40, 30), right=True)
        else:
            self._text(f"Session {env.t} of {env.max_sessions}", self.f_body, MUTED, (42, 62))
            self._pills(ui)

        top, ch = TOP, 378

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
        if ui is not None:
            self._controls(env, ui)
        else:
            self._tutor_choice(env, TUTOR_X, top, TUTOR_W, ch)

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
            self._text(msg, self.f_huge, (255, 255, 255), (W // 2, by + (46 if ui else 59)), center=True)
            if ui:
                self._text("Press N or click New learner to continue", self.f_body, (255, 255, 255), (W // 2, by + 90), center=True)

    def _tutor_choice(self, env, x, top, w, ch):
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

    def _pills(self, ui):
        mouse = pygame.mouse.get_pos()
        labels = {"hint": f"Hint {'ON' if ui['hint_on'] else 'OFF'}", "auto": f"Auto {'ON' if ui['auto'] else 'OFF'}",
                  "track": f"Track: {ui['track_label']}", "new": "New learner"}
        active = {"hint": ui["hint_on"], "auto": ui["auto"], "track": False, "new": False}
        for name, rect in header_pills().items():
            col = BLUE if active[name] else TRACK_BG
            if rect.collidepoint(mouse) and not active[name]:
                col = mix(TRACK_BG, BLUE, 0.25)
            pygame.draw.rect(self.screen, col, rect, border_radius=17)
            self._text(labels[name], self.f_small, (255, 255, 255) if active[name] else TEXT, rect.center, center=True)

    def _controls(self, env, ui):
        # the tutor card in interactive mode, the user clicks the exercise
        x, top, w, ch = TUTOR_X, TOP, TUTOR_W, 378
        self._card(x, top, w, ch, "Your move")
        rects = action_rects()
        live = env.outcome == "running"
        names = ["File", "Word", "Excel", "Safe"]
        for c in range(4):
            self._text(names[c], self.f_small, MUTED, (rects[c * 3].centerx, rects[c * 3].y - 16), center=True)
        for r in range(3):
            self._text(DIFFICULTIES[r], self.f_small, MUTED, (rects[0].x - 8, rects[r].y + 9), right=True)
        for a, rect in rects.items():
            base = DIFF_COLORS[a % 3] if a < 12 else (BLUE if a == 12 else (130, 100, 230))
            chosen = env.last_action == a
            hover = live and ui["hover"] == a
            col = base if chosen else mix(base, (255, 255, 255), 0.45 if hover else 0.78)
            if not live and not chosen:
                col = mix(base, (255, 255, 255), 0.88)
            pygame.draw.rect(self.screen, col, rect, border_radius=9)
            if a >= 12:
                label = "Break" if a == 12 else "Review"
                self._text(label, self.f_body, (255, 255, 255) if chosen else TEXT, rect.center, center=True)
            if ui["hint_on"] and ui["hint"] == a and live:
                pygame.draw.rect(self.screen, TEXT, rect.inflate(6, 6), 3, border_radius=11)
        if ui["hint_on"] and ui["hint"] is not None and live:
            hint = ACTION_NAMES[ui["hint"]].replace(" / ", ", ")
            self._text(f"Agent suggests: {hint}", self.f_small, TEXT, (x + 24, top + 281))
        elif live:
            self._text("Hint is hidden, press H to show it", self.f_small, MUTED, (x + 24, top + 281))
        self._result_pill(env, pygame.Rect(x + 24, top + 300, w - 48, 44))
        self._text(f"Reward  {env.last_reward:+.2f}", self.f_small, MUTED, (x + 24, top + 354))
        self._text(f"Total  {env.total_reward:+.2f}", self.f_small, MUTED, (x + w - 24, top + 354), right=True)

    def _result_pill(self, env, bub):
        a = env.last_action
        if a is None:
            msg, bcol = "Pick the first exercise", TRACK_BG
        elif a < 12:
            ok = env.last_success >= 0.99
            msg, bcol = ("Learner passed", GREEN) if ok else ("Too hard, learner failed", RED)
        elif a == 12:
            msg, bcol = "Learner rested", BLUE
        else:
            msg, bcol = "Quick review done", (130, 100, 230)
        pygame.draw.rect(self.screen, bcol, bub, border_radius=22)
        self._text(msg, self.f_body, TEXT if a is None else (255, 255, 255), bub.center, center=True)

    def close(self):
        pygame.quit()
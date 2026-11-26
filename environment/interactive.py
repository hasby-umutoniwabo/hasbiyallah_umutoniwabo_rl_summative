import pygame

from environment.custom_env import ACTION_NAMES, AdaptiveTutorEnv
from environment.rendering import TRACK_LABELS, TutorRenderer, action_rects, header_pills

KEY_ACTIONS = {pygame.K_b: 12, pygame.K_r: 13}


class InteractiveApp:
    """Play the tutor yourself, or let the trained agent play. Same environment as training."""

    def __init__(self, act, label, track=None, seed=7, speed=2):
        self.act = act
        self.label = label
        self.env = AdaptiveTutorEnv(render_mode=None)
        self.renderer = TutorRenderer("human", 60)
        self.track_mode = 0 if track is None else track + 1  # 0 random, 1 to 3 fixed track
        self.seed = seed
        self.speed = speed
        self.hint_on = True
        self.auto = False
        self.episode = 0
        self.results = []
        self.timer = 0.0
        self.end_wait = 0.0
        self.running = True
        self.start_episode()

    def start_episode(self):
        options = {"track": self.track_mode - 1} if self.track_mode > 0 else None
        self.obs, info = self.env.reset(seed=self.seed + self.episode, options=options)
        self.episode += 1
        self.done = False
        self.timer = self.end_wait = 0.0
        self.renderer.shown = None
        print(f"\nEpisode {self.episode} | learner track: {info['track']}")

    def take(self, action, who):
        if self.done:
            return
        self.obs, reward, term, trunc, info = self.env.step(action)
        print(f"  session {info['sessions']:2d} | {ACTION_NAMES[action]:<24} | reward {reward:+6.2f} | "
              f"motivation {info['motivation']:.2f} | fatigue {info['fatigue']:.2f} | "
              f"readiness {info['weighted_mastery']:.2f} | by {who}")
        if term or trunc:
            self.done = True
            self.results.append((info["outcome"], self.env.total_reward, info["sessions"]))
            print(f"  result: {info['outcome']} after {info['sessions']} sessions, "
                  f"total reward {self.env.total_reward:.2f}")

    def hint(self):
        return None if self.done else self.act(self.obs)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.take(self.act(self.obs), "agent") if not self.done else None
            elif event.key in KEY_ACTIONS:
                self.take(KEY_ACTIONS[event.key], "you")
            elif event.key == pygame.K_a:
                self.auto = not self.auto
            elif event.key == pygame.K_h:
                self.hint_on = not self.hint_on
            elif event.key == pygame.K_n:
                self.start_episode()
            elif event.key == pygame.K_t:
                self.cycle_track()
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.speed = min(8, self.speed + 1)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.speed = max(1, self.speed - 1)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for name, rect in header_pills().items():
                if rect.collidepoint(event.pos):
                    if name == "hint":
                        self.hint_on = not self.hint_on
                    elif name == "auto":
                        self.auto = not self.auto
                    elif name == "track":
                        self.cycle_track()
                    else:
                        self.start_episode()
                    return
            for a, rect in action_rects().items():
                if rect.collidepoint(event.pos):
                    self.take(a, "you")
                    return

    def cycle_track(self):
        # takes effect with the next learner
        self.track_mode = (self.track_mode + 1) % 4
        print(f"  next learners will be: {TRACK_LABELS[self.track_mode]}")

    def update(self, dt):
        if self.auto:
            if self.done:
                self.end_wait += dt
                if self.end_wait > 3.0:
                    self.start_episode()
            else:
                self.timer += dt
                if self.timer >= 1.0 / self.speed:
                    self.timer = 0.0
                    self.take(self.act(self.obs), "agent")

    def ui(self):
        mouse = pygame.mouse.get_pos()
        hover = next((a for a, r in action_rects().items() if r.collidepoint(mouse)), None)
        return {"hint": self.hint(), "hint_on": self.hint_on, "auto": self.auto,
                "track_label": TRACK_LABELS[self.track_mode], "hover": hover}

    def draw(self, dt=1 / 60):
        r = self.renderer
        target = r._targets(self.env)
        if r.shown is None:
            r.shown = dict(target)
        k = min(1.0, 9 * dt)  # bars slide towards the new values
        r.shown = {key: r.shown[key] + (target[key] - r.shown[key]) * k for key in target}
        r._paint(self.env, r.shown, self.ui())
        r._text(f"Agent: {self.label}   |   Click an exercise, or SPACE to follow the agent   |   "
                f"A auto  H hint  N new learner  T track  +/- speed ({self.speed}/s)  Esc quit",
                r.f_small, (112, 124, 148), (40, 616))

    def run(self):
        print(f"Interactive mode with agent: {self.label}")
        clock = pygame.time.Clock()
        while self.running:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                self.handle_event(event)
            self.update(dt)
            self.draw(dt)
            pygame.display.flip()
        if self.results:
            mastered = sum(r[0] == "mastered" for r in self.results)
            avg = sum(r[1] for r in self.results) / len(self.results)
            print(f"\nSummary: {mastered}/{len(self.results)} learners reached workplace readiness, "
                  f"average reward {avg:.2f}")
        pygame.quit()


def run(act, label, track=None, seed=7, speed=2):
    InteractiveApp(act, label, track, seed, speed).run()
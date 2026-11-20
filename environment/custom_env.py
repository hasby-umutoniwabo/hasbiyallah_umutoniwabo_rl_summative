"""AdaptiveTutorEnv - a simulated learner for the Digital Skills Tutor mission.

Mission ("Tech for Good"): close Rwanda's functional digital-skills gap for
university students and graduates in NON-technical fields (health, business /
finance, education).  A web platform must decide, session after session, WHICH
exercise to give a learner next.  Teaching everyone the same material at the
same speed is what the proposal argues against, so the RL agent plays the role
of the adaptive recommender.

Why RL (and not a one-shot classifier):
  * an exercise changes the learner's hidden state (skills, motivation, fatigue)
  * benefits are delayed: a break or an easy review pays off several steps later
  * bad choices have lasting effects (frustration -> the learner drops out)

Learner model (all values in [0, 1]):
  skills[4]   file management, word processing, spreadsheets, safe communication
  motivation  willingness to keep studying; reaching ~0 means drop-out
  fatigue     accumulates with every exercise; reduced by breaks
  track       health | business_finance | education - decides which skills
              matter most for the learner's future workplace
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

SKILLS = ["File mgmt", "Word", "Excel", "Safe comms"]
TRACKS = ["Health", "Business/Finance", "Education"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]
DIFFICULTY_LEVEL = np.array([0.25, 0.55, 0.85], dtype=np.float32)

# How much each skill matters for each track (rows sum to 1).
TRACK_WEIGHTS = np.array(
    [
        [0.30, 0.30, 0.10, 0.30],  # health: records, reports, privacy
        [0.20, 0.20, 0.40, 0.20],  # business / finance: spreadsheets dominate
        [0.20, 0.40, 0.15, 0.25],  # education: documents and communication
    ],
    dtype=np.float32,
)

N_SKILLS = len(SKILLS)
N_EXERCISES = N_SKILLS * len(DIFFICULTIES)  # 12
ACTION_BREAK = N_EXERCISES  # 12
ACTION_REVIEW = N_EXERCISES + 1  # 13 - spaced review of the weakest skill
N_ACTIONS = N_EXERCISES + 2

ACTION_NAMES = [f"{SKILLS[s]} / {DIFFICULTIES[d]}" for s in range(N_SKILLS) for d in range(3)]
ACTION_NAMES += ["Short break", "Review weakest skill"]

OBS_DIM = N_SKILLS + 1 + 1 + 3 + 1 + 1 + N_SKILLS  # 15


def decode_action(a: int):
    """Return (skill_index, difficulty_index) for an exercise action, else None."""
    if a < N_EXERCISES:
        return a // 3, a % 3
    return None


class AdaptiveTutorEnv(gym.Env):
    """One episode = one learner going through the platform (max `max_sessions`)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        max_sessions: int = 60,
        mastery_goal: float = 0.80,
        min_skill_goal: float = 0.55,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_sessions = max_sessions
        self.mastery_goal = mastery_goal
        self.min_skill_goal = min_skill_goal

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

        self._renderer = None
        self.reset_state_only()

    # state
    def reset_state_only(self):
        self.skills = np.zeros(N_SKILLS, dtype=np.float32)
        self.motivation = 0.7
        self.fatigue = 0.0
        self.track = 0
        self.t = 0
        self.since_practice = np.zeros(N_SKILLS, dtype=np.float32)
        self.last_success = 0.5
        self.last_action = None
        self.last_reward = 0.0
        self.last_event = "Learner has just joined"
        self.total_reward = 0.0
        self.outcome = "running"

    def _weighted_mastery(self) -> float:
        return float(np.dot(TRACK_WEIGHTS[self.track], self.skills))

    def _obs(self) -> np.ndarray:
        track_onehot = np.zeros(3, dtype=np.float32)
        track_onehot[self.track] = 1.0
        obs = np.concatenate(
            [
                self.skills,
                [self.motivation, self.fatigue],
                track_onehot,
                [1.0 - self.t / self.max_sessions],
                [self.last_success],
                np.clip(self.since_practice / 20.0, 0.0, 1.0),
            ]
        ).astype(np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _info(self):
        return {
            "weighted_mastery": self._weighted_mastery(),
            "skills": self.skills.copy(),
            "motivation": self.motivation,
            "fatigue": self.fatigue,
            "track": TRACKS[self.track],
            "outcome": self.outcome,
            "sessions": self.t,
        }

    # reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.reset_state_only()
        rng = self.np_random
        self.skills = rng.uniform(0.05, 0.40, size=N_SKILLS).astype(np.float32)
        self.motivation = float(rng.uniform(0.55, 0.85))
        self.track = int(rng.integers(0, 3))
        if options and "track" in options:
            self.track = int(options["track"])
        if self._renderer is not None:
            self.render()
        return self._obs(), self._info()

    # step
    def step(self, action: int):
        action = int(action)
        rng = self.np_random
        prev_mastery = self._weighted_mastery()
        self.t += 1
        self.since_practice += 1.0
        reward = -0.05  # every session costs the learner time
        self.last_action = action
        event = ""

        ex = decode_action(action)
        if ex is not None:
            s, d = ex
            diff = DIFFICULTY_LEVEL[d]
            gap = diff - self.skills[s]  # >0: harder than the learner's level
            # success: logistic in (skill - difficulty), hurt by fatigue, helped by motivation
            logit = 6.0 * (self.skills[s] - diff) + 1.0 + 1.5 * (self.motivation - 0.5) - 2.0 * self.fatigue
            p_success = 1.0 / (1.0 + np.exp(-logit))
            success = bool(rng.random() < p_success)
            self.last_success = 1.0 if success else 0.0

            # zone of proximal development: best learning when slightly above current level
            zpd = np.exp(-((gap - 0.15) ** 2) / (2 * 0.18**2))
            gain = 0.13 * zpd * (0.5 + 0.5 * self.motivation) * (1.0 - 0.5 * self.fatigue)
            gain *= 1.0 if success else 0.45
            self.skills[s] = min(1.0, self.skills[s] + gain)
            self.since_practice[s] = 0.0

            if success:
                self.motivation += 0.03 if gap > -0.1 else -0.01  # too easy = boredom
                event = f"Passed {ACTION_NAMES[action]}"
            else:
                self.motivation -= 0.07 if gap > 0.3 else 0.025  # frustration
                event = f"Failed {ACTION_NAMES[action]}"
            if gap < -0.25:
                self.motivation -= 0.02  # boredom from trivial material
                event += " (too easy)"
            elif gap > 0.35:
                event += " (too hard)"
            self.fatigue += 0.07 + 0.03 * d
        elif action == ACTION_BREAK:
            self.fatigue = max(0.0, self.fatigue - 0.35)
            self.motivation += 0.04
            self.last_success = 0.5
            event = "Learner took a break"
        else:  # ACTION_REVIEW - low-effort spaced review of the weakest skill
            s = int(np.argmin(self.skills))
            self.skills[s] = min(1.0, self.skills[s] + 0.035)
            self.since_practice[s] = 0.0
            self.fatigue += 0.03
            self.motivation += 0.01
            self.last_success = 1.0
            event = f"Reviewed {SKILLS[s]}"

        # forgetting: skills not practised for a while decay slowly
        decay = 0.004 * np.clip(self.since_practice / 10.0, 0.0, 2.0)
        self.skills = np.clip(self.skills - decay, 0.0, 1.0).astype(np.float32)
        self.motivation = float(np.clip(self.motivation, 0.0, 1.0))
        self.fatigue = float(np.clip(self.fatigue, 0.0, 1.0))
        # natural fatigue recovery between sessions
        self.fatigue = max(0.0, self.fatigue - 0.01)

        # reward = learning progress that matters for THIS learner's track
        mastery = self._weighted_mastery()
        reward += 12.0 * (mastery - prev_mastery)
        reward -= 0.15 * max(0.0, self.fatigue - 0.8)  # burn-out penalty

        terminated = False
        truncated = False
        if mastery >= self.mastery_goal and self.skills.min() >= self.min_skill_goal:
            terminated = True
            self.outcome = "mastered"
            reward += 10.0 + 10.0 * (1.0 - self.t / self.max_sessions)  # faster = better
            event = "GOAL: learner is workplace-ready!"
        elif self.motivation <= 0.05:
            terminated = True
            self.outcome = "dropped_out"
            reward -= 10.0
            event = "Learner dropped out"
        elif self.t >= self.max_sessions:
            truncated = True
            self.outcome = "timeout"

        self.last_event = event
        self.last_reward = float(reward)
        self.total_reward += float(reward)
        if self._renderer is not None:
            self.render()
        return self._obs(), float(reward), terminated, truncated, self._info()

    # render
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import TutorRenderer

            self._renderer = TutorRenderer(self.render_mode, self.metadata["render_fps"])
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
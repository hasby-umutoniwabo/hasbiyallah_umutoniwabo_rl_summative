# Adaptive Digital Skills Tutor (RL Summative)

Tech for Good mission: closing Rwanda's functional digital skills gap for university students and graduates in non technical fields (health, business and finance, education).

The RL agent plays the tutor. Each session it chooses the next exercise (or a break) for a simulated learner, trying to get the learner workplace ready in as few sessions as possible without making them quit.

## The environment

`environment/custom_env.py` is a Gymnasium environment, `environment/rendering.py` draws it with Pygame.

- **Actions (14):** 4 modules (file management, Word, Excel, safe communication) x 3 difficulties = 12 exercises, plus a short break and a review of the weakest skill.
- **Observation (15 values in 0 to 1):** 4 skill levels, motivation, fatigue, learner track (one hot, 3 values), time remaining, result of the last exercise, and time since each skill was last practised (4 values).
- **Reward:** progress on the skills that matter for the learner's track, a small time cost per session, a burnout penalty, a bonus for finishing fast and a penalty for dropout.
- **Start state:** random low skills (0.05 to 0.40), random motivation (0.55 to 0.85), random track.
- **Ends when:** the learner is workplace ready (weighted mastery of 0.80 and every skill at least 0.55), the learner drops out (motivation 0.05 or lower), or after 60 sessions.

## Setup and run

Only `uv` is needed.

```
uv sync
uv run main.py
```

This opens the Pygame window and plays the best trained agent on 3 learners, with step by step output in the terminal.

Other options:

```
uv run play.py --algo ppo --episodes 5
uv run play.py --algo dqn --track 1
uv run play.py --random
```

`--track` is 0 for health, 1 for business and finance, 2 for education.

## Interactive mode

Play the tutor yourself. The same environment and trained agents are used.

```
uv run play.py --interactive --algo ppo
```

- Click an exercise (4 skills x 3 levels), Break or Review.
- Hint ON shows what the trained agent would pick. Press SPACE to follow it.
- Auto lets the agent teach by itself, new learners start on their own.
- Keys: A auto, H hint, N new learner, T track, + and - speed, B break, R review, Esc quit.
- Every session is also printed in the terminal, with who chose it (you or the agent).

## Training

Each script runs the hyperparameter sweep for an algorithm (TorchRL).

```
uv run python training/dqn_training.py --all
uv run python training/pg_training.py --algo ppo --all
uv run python training/pg_training.py --algo a2c --all
uv run python training/pg_training.py --algo reinforce --all
```

Use `--run 3` for a single run and `--frames` to change the training length (default 100,000 steps).
Run 11 of A2C was trained for 400,000 steps:

```
uv run python training/pg_training.py --algo a2c --run 11 --frames 400000
```

After training:

```
uv run python training/summarize.py
uv run python training/make_plots.py
```

`summarize.py` tests every run on 100 learners it never saw and picks the best model. `make_plots.py` creates the figures in `assets/plots` and the tables in `assets/tables`.

## Results

Tested on 100 unseen learners, best run per algorithm:

| Algorithm | Best run | Mastery rate | Mean reward |
|---|---|---|---|
| PPO | ppo_03 | 99% | 18.81 |
| A2C (400k steps) | a2c_11 | 100% | 18.01 |
| DQN | dqn_06 | 71% | 11.64 |
| REINFORCE | reinforce_09 | 5% | 2.81 |

All runs except a2c_11 used 100,000 steps. The best A2C run at 100,000 steps reached 8%.
Full logs are in `logs/`, trained models in `models/`.

## Structure

```
main.py             demo of the best agent
play.py             run any trained agent
environment/        custom environment, rendering and interactive mode
training/           DQN, policy gradient training, summary and plots
models/             saved models (dqn, pg)
logs/               training logs and summary.csv
assets/             screenshot, plots and tables
tests/              environment, rendering and interactive checks
```

## Links

- Demo video: [https://youtu.be/tmnOUigceSw](https://youtu.be/tmnOUigceSw)
- Report (PDF): [report/Hasbiyallah_Umutoniwabo_RL_Summative.pdf](report/Hasbiyallah_Umutoniwabo_RL_Summative.pdf)
- Report (Google Doc): [https://docs.google.com/document/d/1vtAayEnK-gXSUtj9KigbXeB2bOpGuGstQ9avJ_Bs8lQ/edit?usp=sharing](https://docs.google.com/document/d/1vtAayEnK-gXSUtj9KigbXeB2bOpGuGstQ9avJ_Bs8lQ/edit?usp=sharing)

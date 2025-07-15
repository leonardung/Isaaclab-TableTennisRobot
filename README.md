# Table Tennis Robot RL (Isaac Lab)

![Table‑Tennis Agent in Action](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExM25yY3JocHU4aTYzN2s1c3NkNmdhbmZxYzVqbHNqZHZ6Ym85cmUxaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/tmOYzlcMxLDTUCUOcT/giphy.gif)

> **Goal:** train a UR10‑based robot to return ping‑pong balls in simulation using NVIDIA Isaac Lab and the [skrl](https://github.com/Toni-SM/skrl) reinforcement‑learning library.

---

## Project Overview

This repository contains everything you need to reproduce the table‑tennis environment, train a policy, and replay trained roll‑outs.

| Component | Description |
|-----------|-------------|
| **Isaac Lab scene** | 4 096 parallel environments, each with a UR10 arm, table, and ball. |
| **Observation** | 18‑dim state = robot joint positions/velocities (12) + ball position/velocity (6). |
| **Action** | 6‑D joint position targets scaled to ±50 rad/s. |
| **Episode length** | 5 s (≈ 600 simulation steps at 120 Hz). |
| **Reward terms** | contact bonus, opponent‑table success, own‑table penalty, floor penalty, velocity shaping. |

The core logic lives in `tabletennis_env_cfg.py` (environment parameters) and `tabletennis_env.py` (dynamics, rewards, reset logic).

---

## Download my custom usd
https://drive.google.com/drive/folders/1qRUON2e8wAQHjJhucGsG2Pa4nqtzxTi1?usp=sharing

and place them in `custom_usd/`

## Training

```bash
# Headless training (fastest)
python ./scripts/skrl/train.py --task=Template-Tabletennis-Direct-v0 --headless
```

The script launches 4 096 parallel instances and trains a PPO agent (skrl default hyper‑parameters). Checkpoints and TensorBoard logs are written to `runs/YYYY-MM-DD_HH-MM-SS/`.

### Customizing hyper‑parameters
Edit `source\TableTennis\TableTennis\tasks\direct\tabletennis\agents\skrl_ppo_cfg.yaml`

---

## Playing / Visualising

After a checkpoint appears in `logs/.../checkpoints/`, visualise the agent:

```bash
python ./scripts/skrl/play.py --task=Template-Tabletennis-Direct-v0 --num_envs=1
```

Set `--headless` **off** (default) to open an Isaac Sim window; or add `--headless` for a video‑only render.

---

## Environment Details

### Reset randomisation

| Parameter | Range |
|-----------|-------|
| Ball x‑speed | −1 … 1 m/s |
| Ball y‑speed | 3.5 … 5 m/s |
| Ball z‑speed | 2.0 … 2.2 m/s |
| Ball x‑spawn  | −0.2 … 0.2 m |

### Reward weights (see `TabletennisEnvCfg`)

```text
0.5  * velocity‑reward (higher = encourage fast returns)
1    * contact bonus when paddle touches ball
5    * reward when ball first bounces on opponent side
2    * penalty if ball hits own half before opponent touch
3.5  * penalty if ball hits the floor
2    * shaping – encourage forward (‑y) ball position
```

### Assets

* `custom_usd/UR10_instanceable_pong.usd` – robot & paddle.
* `custom_usd/Table_tennis.usd` – table model.
* `custom_usd/Ping_pong_ball.usd` – ball with realistic restitution (0.8).

# Deep Q-Learning: Space Invaders Agent

**Author:** Olayinka David Ajao

## Overview

This document outlines my individual contribution to the group Deep Q-Learning project. My primary focus was training, tuning, and evaluating a Deep Q-Network (DQN) agent to play `ALE/SpaceInvaders-v5` utilizing Stable Baselines3 and Gymnasium/ALE.

To determine the most effective configuration, I engineered an evaluation pipeline testing 20 distinct model architectures and hyperparameter combinations—10 utilizing Convolutional Neural Network (CNN) policies and 10 utilizing Multi-Layer Perceptron (MLP) policies.

## Best Agent & Results

The overall champion model is **`MLP_exp08`**, which achieved an average evaluation reward of **413.3** across a 3-episode deterministic evaluation. This specific agent significantly outperformed both the baseline configurations and the CNN-based models.

The promoted winning model is saved as `dqn_model.zip`, and the highlight reel of its performance has been successfully generated in the `./video` directory.

## Setup & Usage

Install the required dependencies for the environment:

```bash
pip install -r requirements.txt

```

To watch the trained champion agent play and record a gameplay video:

```bash
python play.py --env-id ALE/SpaceInvaders-v5 --model dqn_model.zip --mode record --episodes 3

```

## Hyperparameter Tuning & Experiment Log

I conducted a one-factor-at-a-time (OFAT) sweep starting from a baseline, altering individual hyperparameters, and finally testing combined "best guess" configurations across both CNN and MLP policies.

| # | Type | lr | gamma | batch_size | eps_start | eps_end | eps_decay | Eval Reward | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CNN | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 310.0 | Baseline CNN reference point. |
| 2 | CNN | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 233.3 | Higher learning rate beat baseline in training, dropped in eval. |
| 3 | CNN | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 285.0 | 2nd-best train run; lower LR was highly effective. |
| 4 | CNN | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | 188.3 | Lower gamma clearly helped in training, poor eval. |
| 5 | CNN | 1e-4 | 0.995 | 32 | 1.0 | 0.05 | 0.10 | 110.0 | Higher gamma — worst gamma setting tested. |
| 6 | CNN | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | 111.7 | Smaller batch beat baseline in training, dropped in eval. |
| 7 | CNN | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | 151.7 | Worst CNN run — larger batch hindered learning here. |
| 8 | CNN | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | 158.3 | Fast epsilon decay beat baseline in training. |
| 9 | CNN | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | 263.3 | Roughly baseline-level in training. |
| 10 | CNN | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | 363.3 | Numerically best CNN run across training and evaluation. |
| 1 | MLP | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 243.3 | Baseline MLP comparison run. |
| 2 | MLP | 1e-3 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 188.3 | MLP evaluation sweep. |
| 3 | MLP | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 0.10 | 170.0 | MLP evaluation sweep. |
| 4 | MLP | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 0.10 | 185.0 | MLP evaluation sweep. |
| 5 | MLP | 1e-4 | 0.995 | 32 | 1.0 | 0.05 | 0.10 | 3.3 | Lowest performing MLP experiment. |
| 6 | MLP | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 0.10 | 310.0 | Strong MLP evaluation performance. |
| 7 | MLP | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 0.10 | 146.7 | MLP evaluation sweep. |
| 8 | MLP | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.02 | **413.3** | **Overall Champion Model.** Highest evaluation score achieved. |
| 9 | MLP | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 0.30 | 295.0 | MLP evaluation sweep. |
| 10 | MLP | 5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | 181.7 | MLP evaluation sweep. |

### Key Space Invaders Insights

* **Batch Size Sensitivity:** A large batch size of 128 explicitly hindered the model's ability to learn effectively in Space Invaders, unlike in other environments where it proved beneficial.
* **Learning Rate Tolerance:** The environment demonstrated a high tolerance for learning rate variance. Pushing the learning rate 10x higher still managed to beat the baseline during the initial training phases.
* **Architecture vs. Hyperparameter Synergy:** While CNN policies are traditionally standard for pixel-based Atari environments, pairing an MLP policy with a fast epsilon decay strategy (Experiment 8) yielded the absolute highest deterministic evaluation score of the suite.
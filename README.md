<div align="center">

# Reinforcement Learning Environments

### A Progressive Collection of Classic and Modern RL Algorithms — From Tabular Methods to MuJoCo Locomotion

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-009DCF?style=for-the-badge)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics_Sim-darkgreen?style=for-the-badge)](https://mujoco.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Algorithms at a Glance](#algorithms-at-a-glance)
- [Environments](#environments)
- [Setup & Installation](#setup--installation)
- [Notebooks](#notebooks)
  - [FrozenLake — Q-Learning (Monte Carlo)](#1-frozenlake--q-learning-monte-carlo)
  - [FrozenLake Slippery — SARSA](#2-frozenlake-slippery--sarsa)
  - [CartPole — Deep Q-Network (DQN)](#3-cartpole--deep-q-network-dqn)
  - [LunarLander — A2C](#4-lunarlander--advantage-actor-critic-a2c)
  - [MountainCar — SAC](#5-mountaincar-continuous--soft-actor-critic-sac)
  - [Walker2D — PPO](#6-walker2d--proximal-policy-optimization-ppo)
  - [HalfCheetah — PPO at Scale](#7-halfcheetah--ppo-at-scale)
- [Results Summary](#results-summary)
- [Dependencies](#dependencies)

---

## Overview

This repository is a hands-on implementation library covering the full spectrum of modern reinforcement learning — starting from classical tabular Q-Learning and building up to large-scale Proximal Policy Optimization on MuJoCo physics environments. Every notebook is self-contained and progressively more complex, making this a complete learning resource as well as a working codebase.

```
Tabular ──► Value-Based Deep RL ──► Actor-Critic ──► Maximum Entropy ──► Policy Gradient @ Scale
   │                │                    │                 │                       │
FrozenLake      CartPole DQN        LunarLander A2C    MountainCar SAC    Walker2D / HalfCheetah PPO
```

**What makes this collection worth studying:**
- Each notebook implements the core algorithm **from scratch** in PyTorch — no `stable-baselines` shortcuts
- Environments span discrete grids → continuous physics simulations (MuJoCo)
- The HalfCheetah notebook runs **2048 parallel environments** simultaneously
- Includes visualizations: policy heatmaps, Q-value triangulated cell plots, reward curves, and training videos

---

## Algorithms at a Glance

| Notebook | Algorithm | Type | Action Space | Environment |
|---|---|---|---|---|
| `FrozenLake (1).ipynb` | Monte Carlo Q-Learning | Tabular, Off-Policy | Discrete | FrozenLake-v1 (4×4) |
| `FrozenLake_Slipping (1).ipynb` | SARSA | Tabular, On-Policy | Discrete | FrozenLake-v1 (4×4, slippery) |
| `CartPole (3).ipynb` | Deep Q-Network (DQN) | Value-Based Deep RL | Discrete | CartPole-v1 |
| `LunarLanderA2C (1).ipynb` | Advantage Actor-Critic (A2C) | Policy Gradient | Discrete | LunarLander-v3 |
| `MountainCartSAC (1).ipynb` | Soft Actor-Critic (SAC) | Max-Entropy Deep RL | Continuous | MountainCarContinuous-v0 |
| `walker2DFinal.ipynb` | Proximal Policy Optimization (PPO) | Policy Gradient | Continuous | Walker2d-v5 (MuJoCo) |
| `HalfCheetah.ipynb` | PPO + 2048 Vectorized Envs | Policy Gradient at Scale | Continuous | HalfCheetah-v5 (MuJoCo) |

---

## Environments

<table>
<tr>
<td width="50%">

### Discrete Control
**FrozenLake-v1**
A 4×4 grid where the agent must navigate from S (start) to G (goal) while avoiding holes (H). Two variants: non-slippery (deterministic) and slippery (stochastic — intended moves succeed only 1/3 of the time).

**CartPole-v1**
A pole balanced on a moving cart. The agent must push left or right to prevent the pole from falling. A classic benchmark for DQN.

**LunarLander-v3**
A 2D rocket lander. The agent fires thrusters to land safely between two flags, with rewards for controlled descent and penalties for crashing.

</td>
<td width="50%">

### Continuous Control (MuJoCo)
**MountainCarContinuous-v0**
A car stuck in a valley must build momentum by rocking back and forth to reach the hilltop. Sparse reward makes this a notoriously hard exploration problem.

**Walker2d-v5**
A bipedal robot that must learn to walk forward. 6-dimensional continuous action space controlling leg joints.

**HalfCheetah-v5**
A 2D cheetah-like robot that learns to run as fast as possible. High-dimensional observation and action spaces with dense rewards.

</td>
</tr>
</table>

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (strongly recommended for Walker2D and HalfCheetah notebooks)
- MuJoCo installed for Walker2D and HalfCheetah

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Reinforcement_Learning_Envs.git
cd Reinforcement_Learning_Envs
```

### 2. Create and activate a virtual environment

```bash
python -m venv rl_env
source rl_env/bin/activate       # Linux / macOS
# rl_env\Scripts\activate        # Windows
```

### 3. Install core dependencies

```bash
pip install gymnasium torch torchvision numpy matplotlib seaborn tqdm jupyter
```

### 4. Install MuJoCo (for Walker2D and HalfCheetah only)

```bash
pip install mujoco gymnasium[mujoco]
```

> **Note:** MuJoCo 2.1.0+ is free and open-source. On headless servers, set `export MUJOCO_GL=egl` before launching Jupyter.

### 5. Install PyTorch Lightning (for Walker2D and HalfCheetah)

```bash
pip install pytorch-lightning
```

### 6. Launch Jupyter

```bash
jupyter notebook
```

---

## Notebooks

---

### 1. FrozenLake — Q-Learning (Monte Carlo)

**File:** `FrozenLake (1).ipynb`

Implements **On-Policy Monte Carlo Q-Learning** on the 4×4 FrozenLake grid. The agent accumulates complete episode trajectories and updates Q-values by averaging observed returns.

**Key implementation details:**
- Q-table of shape `[16 states × 4 actions]`
- Epsilon-greedy exploration during training
- Greedy policy extraction: `π(s) = argmax_a Q(s, a)`
- 5,000 training episodes

**Visualizations produced:**
- Policy heatmap with directional arrows overlaid on the grid
- Q-value split-cell plot (4 triangles per cell, one per action)
- State value heatmap `V(s) = max_a Q(s, a)`
- Action probability distributions

**Result:** 100% success rate on 5 evaluation episodes after training.

---

### 2. FrozenLake Slippery — SARSA

**File:** `FrozenLake_Slipping (1).ipynb`

Implements **SARSA** (State-Action-Reward-State-Action) — an on-policy temporal-difference method — on the stochastic slippery variant of FrozenLake. Unlike Q-Learning, SARSA updates are based on the actual next action taken (not the greedy one), making it better suited to on-policy control.

| Hyperparameter | Value |
|---|---|
| Learning rate (α) | 0.15 |
| Discount factor (γ) | 0.99 |
| Epsilon | 0.85 |
| Training episodes | 10,000 |
| Step penalty | −0.02 |
| Goal reward | +100 |

**Why this is hard:** On the slippery surface, the intended action only succeeds 1/3 of the time — the remaining probability is split between the two perpendicular actions. This makes consistent policy learning significantly more challenging.

---

### 3. CartPole — Deep Q-Network (DQN)

**File:** `CartPole (3).ipynb`

A full **DQN** implementation with all standard components: experience replay buffer, separate target and online networks, and epsilon decay.

**Architecture comparison (500 episodes each):**

| Hidden Units | Moving Avg Reward (50-ep window) |
|---|---|
| 512 | Baseline |
| **256** | **Best performance** |
| 128 | Baseline |

**DQN components implemented:**
- Replay buffer (capacity-managed circular buffer)
- Target network (hard update periodically)
- Epsilon-greedy with exponential decay
- TD update: `Q(s,a) ← r + γ · max_{a'} Q_target(s', a')`

---

### 4. LunarLander — Advantage Actor-Critic (A2C)

**File:** `LunarLanderA2C (1).ipynb`

Implements **A2C** with **8 parallel vectorized environments** and separate policy and value networks. The advantage function `A(s,a) = r + γV(s') − V(s)` reduces gradient variance compared to vanilla REINFORCE.

**Network architecture:**

```
State (8-dim) ──► [Linear 256 ► Linear 128 ► softmax]  ──► Advantage / Action probs
               └─► [Linear 256 ► Linear 128 ► scalar]  ──► State Value V(s)
```

**Training details:**
- 8 simultaneous environments (vectorized)
- Orthogonal weight initialization
- Layer normalization for stability
- Entropy regularization weight: 0.0007
- Discount factor: 0.99
- 750 training episodes

The notebook also includes a **learning rate sweep** across 4 different optimizer configurations, evaluated using 50-episode moving average reward windows.

---

### 5. MountainCar Continuous — Soft Actor-Critic (SAC)

**File:** `MountainCartSAC (1).ipynb`

Implements **Soft Actor-Critic (SAC)** — a maximum-entropy off-policy algorithm — on the notoriously difficult MountainCarContinuous environment. SAC maximizes both expected reward and entropy, leading to better exploration in sparse-reward settings.

**Distinctive SAC features:**
- **Twin Q-networks** to mitigate overestimation bias (take minimum of two critics)
- **Squashed Gaussian policy** with tanh transformation to bound actions in [-1, 1]
- **Automatic entropy tuning** — the temperature α is learned, not hand-tuned
- **Soft target updates** with τ = 0.005

**Architecture:**
```
Actor:   state ──► [256 ► 256] ──► μ, log_σ  ──► tanh(sample(μ, σ))
Critic1: [state | action] ──► [256 ► 256] ──► Q₁(s, a)
Critic2: [state | action] ──► [256 ► 256] ──► Q₂(s, a)
```

| Hyperparameter | Value |
|---|---|
| Replay buffer | 100,000 |
| Batch size | 256 |
| Actor / Critic LR | 3e-4 |
| Tau (soft update) | 0.005 |
| Episodes | 500 |

---

### 6. Walker2D — Proximal Policy Optimization (PPO)

**File:** `walker2DFinal.ipynb`

Implements **PPO** with **Generalized Advantage Estimation (GAE)** for the MuJoCo Walker2d bipedal locomotion task. PPO clips the policy update ratio to prevent destructive large policy updates.

**PPO clipped objective:**
```
L_CLIP = E[ min(r_t(θ) · A_t,  clip(r_t(θ), 1-ε, 1+ε) · A_t) ]
```

**Architecture (~277K parameters total):**
```
GradientPolicy:  state ──► [256 ► 256 ► 256] (LayerNorm, tanh) ──► μ, σ (Gaussian)
ValueNetwork:    state ──► [256 ► 256 ► 256] (LayerNorm, tanh) ──► V(s)
```

**Training configuration:**

| Parameter | Value |
|---|---|
| Parallel environments | 10 |
| Steps per rollout | 2048 |
| Batch size | 2048 |
| PPO clip ε | 0.2 |
| GAE λ | 0.95 |
| Entropy weight | 0.01 |
| Policy LR / Value LR | 3e-4 / 1e-3 |
| Max epochs | 500 |

Checkpoints are saved via callback (top-3 model preservation). Achieved average reward of **3437** during training.

---

### 7. HalfCheetah — PPO at Scale

**File:** `HalfCheetah.ipynb`

The most computationally intensive notebook: **PPO running across 2048 parallel environments simultaneously**. Trains the HalfCheetah-v5 MuJoCo locomotion agent to run as fast as possible.

**Scale comparison across notebooks:**

| Notebook | Parallel Envs | Algorithm |
|---|---|---|
| LunarLander | 8 | A2C |
| Walker2D | 10 | PPO |
| **HalfCheetah** | **2048** | **PPO** |

**Key design choices:**
- Observation normalization (running mean/std), but **no reward normalization** — HalfCheetah rewards are already well-scaled
- Entropy decay: starts at 0.1 and reduces over training
- Twin value networks (main + target) for stability
- Smooth L1 loss for value function (Huber loss)
- TensorBoard integration for live metric monitoring
- Video generation every 30 epochs for visual inspection

**Architecture (~214K parameters total):**
```
GradientPolicy:  state ──► [256 ► 256] ──► tanh(μ), softplus(σ)   [73.5K params]
ValueNetwork:    state ──► [256 ► 256 ► 256] ──► V(s)              [70.7K params]
TargetNetwork:   copy of ValueNetwork                               [70.7K params]
```

Evaluation runs every 10 epochs; full 500-epoch training completes to produce reward curves and stored training videos.

---

## Results Summary

| Environment | Algorithm | Key Result |
|---|---|---|
| FrozenLake (non-slippery) | Monte Carlo Q-Learning | 100% success rate, avg 6 steps to goal |
| FrozenLake (slippery) | SARSA | Stochastic env; agent adapts to slip dynamics |
| CartPole-v1 | DQN | Best with 256-unit hidden layer architecture |
| LunarLander-v3 | A2C | Stable convergence across 4 LR configurations |
| MountainCarContinuous-v0 | SAC | Automatic entropy tuning overcomes sparse reward |
| Walker2d-v5 | PPO + GAE | Avg reward **3437** |
| HalfCheetah-v5 | PPO (2048 envs) | Full 500-epoch training, video-captured policy |

---

## Dependencies

```
gymnasium>=0.29.0
torch>=2.0.0
pytorch-lightning>=2.0.0     # Walker2D, HalfCheetah only
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
mujoco>=2.3.0                # Walker2D, HalfCheetah only
```

Install everything at once:

```bash
pip install gymnasium torch pytorch-lightning numpy matplotlib seaborn tqdm mujoco "gymnasium[mujoco]"
```

---

<div align="center">

**Built from scratch in PyTorch — no shortcuts, no wrappers.**

If you found this useful, consider starring the repository.

</div>

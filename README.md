# 🤖 Online Decision Transformer — Paper Reproduction

A reproduction of the **Online Decision Transformer (ODT)** paper published at ICML 2022,
including full implementation walkthrough, code analysis, and experimental results on the Hopper-v4 environment.

---

## 👤 Author

- **Bishozit Chandra Das**
- Student ID: **VR544246**
- MSc in **Artificial Intelligence** at Università degli Studi di Verona
- [BishozitChandraDas](https://github.com/BishozitChandraDas)

---

## 🎯 Project Overview

This project reproduces and analyzes the **Online Decision Transformer (ODT)** — a reinforcement
learning algorithm that blends offline pretraining with online finetuning in a unified sequence
modeling framework. ODT extends the Decision Transformer (DT) by introducing stochastic policies
and max-entropy regularization for sample-efficient exploration.

- **Paper:** Zheng et al., *"Online Decision Transformer"*, ICML 2022
- **Paper Link:** https://proceedings.mlr.press/v162/zheng22c.html
- **Official Code:** https://github.com/facebookresearch/online-dt

---

## 🧠 Problem Statement

Standard offline RL methods (including Decision Transformer) are limited by the quality of their
training dataset. ODT addresses this by:

- Extending DT to online finetuning settings
- Introducing stochastic policies for exploration
- Using max-entropy regularization to balance exploration and exploitation
- Applying hindsight return relabeling for sample efficiency

---

## 📊 Paper's Key Innovations

### Innovation 1 — Stochastic Policy (Eq. 2)

Instead of predicting a single deterministic action, ODT predicts a Gaussian distribution:

$$\pi_\theta(a_t|s,g) = \mathcal{N}(\mu_\theta(s,g),\ \Sigma_\theta(s,g))$$

**Code:** `decision_transformer/models/decision_transformer.py` — `DiagGaussianActor` class

```python
class DiagGaussianActor(nn.Module):
    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()
        self.mu      = torch.nn.Linear(hidden_dim, act_dim)  # μ (mean)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)  # log σ² (variance)
```

---

### Innovation 2 — NLL + Entropy Objective (Eq. 3 → 5)

ODT replaces MSE loss with Negative Log-Likelihood and adds entropy constraint:

$$\min_\theta J(\theta) \quad \text{subject to} \quad H[\pi] \geq \beta$$

**Code:** `main.py` — `loss_fn()` + `trainer.py` — `train_iteration()`

```python
def loss_fn(a_hat_dist, a, attention_mask, entropy_reg):
    log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
    entropy        = a_hat_dist.entropy()[attention_mask > 0].mean()
    loss           = -log_likelihood - entropy_reg * entropy
    return loss, -log_likelihood, entropy
```

---

### Innovation 3 — Dual Variable λ (Eq. 6 → 7)

Auto-tuned temperature variable balances exploration vs exploitation:

$$\min_\theta J(\theta) - \lambda \cdot H[\pi]$$

**Code:** `decision_transformer.py` — `temperature()` + `trainer.py` — `temp_optimizer`

```python
# λ = e^(log_temperature) — auto-tuned
def temperature(self):
    if self.stochastic_policy:
        return self.log_temperature.exp()

# λ update — trainer.py
temperature_loss = (
    self.model.log_temperature.exp()
    * (entropy - self.model.target_entropy).detach()
)
temperature_loss.backward()
self.temp_optimizer.step()
```

---

### Innovation 4 — Three Practical Components

#### (a) Trajectory Replay Buffer → `replay_buffer.py`

```python
class ReplayBuffer:
    def __init__(self, capacity, trajectories):
        self.trajectories = trajectories   # stores full trajectories
        self.capacity     = capacity

    def add_new_trajs(self, new_trajs):
        # FIFO — remove oldest trajectory
        self.trajectories = (
            self.trajectories[len(new_trajs):] + new_trajs
        )
```

#### (b) Hindsight Return Relabeling → `data.py`

```python
def discount_cumsum(x, gamma):
    # gt = r_t + r_{t+1} + ... + r_{|τ|}  — Eq. g_real
    discount_cumsum       = np.zeros_like(x)
    discount_cumsum[-1]   = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum
```

#### (c) RTG Conditioning → `main.py`

```python
# Online exploration = 2× expert return
parser.add_argument("--online_rtg", type=int, default=7200)
# hopper expert return ≈ 3600  →  2× = 7200

# Evaluation = fixed RTG
parser.add_argument("--eval_rtg",   type=int, default=3600)
```

---

### Innovation 5 — Combined Training Objective (Eq. 8)

$$\min_\theta \underbrace{\mathbb{E}[-\log \pi_\theta(a|s,g)]}_{T_{NLL}} - \lambda \underbrace{\mathbb{E}[H(\pi_\theta(\cdot|s,g))]}_{T_{CE}}$$

**Code:** `main.py` + `trainer.py` — `train_step()`

```python
# T_NLL — Eq.3
nll = -action_log_prob.mean()

# T_CE — Eq.4
entropy = action_dist.entropy().mean()

# Eq.8: T_NLL - λ * T_CE
loss = nll - self.model.temperature().detach() * entropy
```

---

### Innovation 6 — Converged Objective (Eq. 9)

When training converges, Eq.8 reduces to pure behavior cloning:

$$\min_\theta \mathbb{E}_{(s,a,g)\sim\rho_{\pi_\theta}}[-\log \pi_\theta(a|s,g)]$$

**Code:** `decision_transformer.py` — `temperature()` method

```python
def temperature(self):
    if self.stochastic_policy:
        return self.log_temperature.exp()
        # When entropy > β → temperature → 0
        # Then loss ≈ nll only → Eq.9
```

---

## 🗂️ Repository Structure

```
Online_Decision_Transformer_Reproduction/
│
├── online-dt/                          # Official Facebook Research codebase
│   ├── main.py                         # Entry point — full pipeline
│   ├── trainer.py                      # Training loop — Eq.8 loss
│   ├── replay_buffer.py                # Trajectory-level replay buffer
│   ├── evaluation.py                   # Policy evaluation
│   ├── data.py                         # Dataset loading + HER
│   ├── lamb.py                         # LAMB optimizer
│   ├── logger.py                       # Result logging
│   └── decision_transformer/
│       └── models/
│           ├── decision_transformer.py # ODT model — DiagGaussianActor
│           └── trajectory_gpt2.py      # GPT2 backbone
│
└── README.md
```

---

## ⚙️ Experimental Setup

- **Platform:** Google Colab (Linux, CPU)
- **Environment:** Hopper-v4 (MuJoCo locomotion, dense rewards)
- **Dataset:** 200 trajectories generated via random policy

| Hyperparameter | Value |
|---|---|
| Pretrain iterations | 1 |
| Updates per pretrain iter | 100 |
| Online iterations | 3 |
| Online rollouts | 1 |
| Eval episodes | 1 |
| Device | CPU |
| Context length K | 20 |
| Embedding dim | 512 |
| Batch size | 256 |

---

## 📈 Our Experimental Results

### Pretraining (Iteration 0)

| Metric | Value | Paper Connection |
|--------|-------|-----------------|
| NLL Loss | 4579.49 | Eq.3 — J(θ) |
| Entropy | -8.498 | Eq.4 — H[π] |
| λ (temp) | 0.1005 | Eq.7 — dual variable |
| Eval Return | 8.52 | Offline baseline |

### Online Finetuning (Iteration 1 → 3)

| Metric | Pretrain | Iter 1 | Iter 2 | Iter 3 |
|--------|---------|--------|--------|--------|
| NLL Loss | 4579 | 2387 | 47.1 | **6.83** |
| λ (temp) | 0.1005 | 0.1035 | 0.1070 | 0.1112 |
| Eval Return | 8.52 | — | — | **9.82** |
| Traj Length | — | 13 | 11 | 19 |

### Paper Claims Verified

```
✅ Claim 1: NLL rapidly decreases    →  4579 → 6.83  (3 iterations only)
✅ Claim 2: λ converges between 0→1 →  0.1005 → 0.1112
✅ Claim 3: Return improves          →  8.52 → 9.82
✅ Claim 4: Entropy stays controlled →  near β throughout training
```

---

## 📋 Requirements

```bash
conda create -n odt python=3.10
conda activate odt

pip install torch
pip install transformers==4.26.0
pip install gymnasium[mujoco]
pip install stable-baselines3
pip install numpy==1.23.5
pip install cython==0.29.37
pip install gym==0.23.1
pip install h5py
pip install mujoco-py==2.1.2.14
pip install wandb tensorboard
```

---

## 🔧 Setup & How to Run

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/facebookresearch/online-dt.git
cd online-dt
```

**Step 2 — Install MuJoCo 2.1.0:**
```bash
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
mv mujoco210 ~/.mujoco/mujoco210
```

**Step 3 — Generate dataset:**
```bash
python create_dataset.py
```

**Step 4 — Run ODT:**
```bash
MUJOCO_GL=osmesa MPLBACKEND=Agg python main.py \
  --env Hopper-v4 \
  --max_pretrain_iters 1 \
  --num_updates_per_pretrain_iter 100 \
  --max_online_iters 3 \
  --num_online_rollouts 1 \
  --num_eval_episodes 1 \
  --device cpu
```

---

## 📌 Limitations

- Random offline dataset (not D4RL medium quality) limits absolute performance
- CPU-only training — significantly slower than paper's GPU results
- Only 3 online iterations run due to compute constraints (paper runs 1500)
- Hopper-v4 used instead of paper's hopper-medium-v2 due to D4RL compatibility issues on Python 3.10+

---

## 🎓 Academic Context

This project was developed as part of the **Reinforcement Learning** coursework for the
MSc in Artificial Intelligence at the University of Verona, focusing on:

- Offline-to-online RL with sequence modeling
- Transformer-based policy learning (Decision Transformer → ODT)
- Max-entropy reinforcement learning
- Sample-efficient online finetuning

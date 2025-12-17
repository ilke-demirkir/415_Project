# 2048 Policy Gradient Agent

Train and play 2048 with a lightweight policy‑gradient agent built in PyTorch. The repository includes a minimal 2048 environment, a pygame front end for manual play, and a policy network that learns via REINFORCE with a running baseline, entropy bonus, and simple corner‑based reward shaping.

## Project Layout
- `Policy_Gradient.py`: Policy network, training loop, evaluation routine, and plotting helpers.
- `Game_2048_BE.py`: 2048 environment with move simulation, reward shaping, and game‑over detection.
- `Game_2048_FE.py`: Pygame front end to play 2048 manually with the keyboard.

## Setup
1) Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
2) Install dependencies:
   ```bash
   pip install torch numpy matplotlib pygame
   ```
   If you need a platform‑specific PyTorch build (e.g., for Apple Silicon GPU), follow the install command from https://pytorch.org for your system and then install the remaining packages.

## Training the Agent
Run the training script; it auto‑detects CUDA if available:
```bash
python Policy_Gradient.py
```
Key parameters (see `train_reinforce` in `Policy_Gradient.py`):
- `num_episodes`: training episodes (default 1000 in the `__main__` block).
- `lr`, `gamma`, `entropy_coef`, `baseline_momentum`, `max_steps_per_episode`.

During training, the script prints per‑episode scores and a moving average. After training it plots the average training score, evaluates the policy for a fixed number of episodes, and plots evaluation scores.

### Saving Plots
Both `plot_average_score` and `plot_evaluation_scores` accept a `save_path` argument if you want to write figures instead of displaying them:
```python
plot_average_score(scores, window=100, smoothing_factor=0.9, save_path="train_scores.png")
plot_evaluation_scores(eval_scores, window=10, smoothing_factor=0.9, save_path="eval_scores.png")
```

## Evaluating a Trained Policy
To run evaluation separately (for example, after loading a saved model you add later):
```python
from Policy_Gradient import PolicyNet, evaluate_policy
import torch

policy = PolicyNet()
policy.load_state_dict(torch.load("policy.pt", map_location="cpu"))
evaluate_policy(policy, num_episodes=50, device="cpu")
```

## Playing Manually
Launch the pygame front end and use the arrow keys:
```bash
python Game_2048_FE.py
```
This uses the same backend environment as training, so merges, spawns, and game‑over logic match the RL setup.

## Notes and Tips
- The environment shapes rewards with a corner heuristic, a small penalty for no‑op moves, and scaled merge rewards (`merge_scale`) to stabilize learning.
- Episodes are capped by `max_steps_per_episode` during training to avoid degenerate stuck policies.
- If you want to checkpoint models, add `torch.save(policy_net.state_dict(), "policy.pt")` after training completes.

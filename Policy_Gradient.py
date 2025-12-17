# policy_pg_2048.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from Game_2048_BE import Game2048Env  # import the env above


# ---- State encoding: board (4x4) -> 16-dim vector ----

def encode_state(board):
    """
    board: 4x4 list of ints (0, 2, 4, 8, ...)
    We map tiles to log2(value) / 16 for nicer ranges.
    0 -> 0.0
    2 -> 1/16
    4 -> 2/16
    ...
    """
    flat = []
    for i in range(4):
        for j in range(4):
            v = board[i][j]
            if v == 0:
                flat.append(0.0)
            else:
                flat.append(np.log2(v) / 16.0)
    return np.array(flat, dtype=np.float32)


# ---- Policy Network ----

class PolicyNet(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch_size, 16)
        return self.net(x)  # logits for 4 actions


def select_action(policy_net, state, device="cpu"):
    """
    Given a board state, sample an action from the policy.
    Returns: action (int), log_prob (tensor), entropy (tensor)
    """
    state_vec = encode_state(state)
    state_tensor = torch.from_numpy(state_vec).to(device).unsqueeze(0)  # (1,16)

    logits = policy_net(state_tensor)  # (1,4)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    return action.item(), log_prob, entropy

# continue in policy_pg_2048.py

def compute_returns(rewards, gamma=0.99):
    """
    rewards: list of r_t for one episode
    returns: list of G_t (same length)
    """
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train_reinforce(
    num_episodes=2000,
    gamma=0.99,
    lr=1e-3,
    print_every=50,
    device="cpu",
    baseline_momentum=0.05,
    entropy_coef=0.0005,      # try smaller first
    max_steps_per_episode=2000,
):
    env = Game2048Env()
    policy_net = PolicyNet().to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    all_scores = []
    baseline = 0.0
    best_score = float("-inf")
    best_tile = 0

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        log_probs = []
        rewards = []
        entropy_terms = []
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            action, log_prob, entropy = select_action(policy_net, state, device=device)

            next_state, reward, done, info = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            entropy_terms.append(entropy)

            state = next_state
            steps += 1

        score = info["score"]
        all_scores.append(score)
        best_score = max(best_score, score)
        episode_max_tile = max(max(row) for row in env.board)
        best_tile = max(best_tile, episode_max_tile)

        # Returns
        returns = compute_returns(rewards, gamma=gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Episode-level baseline update (EMA of mean return)
        G_mean = returns.mean().item()
        baseline = (1 - baseline_momentum) * baseline + baseline_momentum * G_mean

        # Advantages
        advantages = returns - baseline

        # Optional normalization (usually helpful)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Loss
        log_probs_t = torch.stack(log_probs)
        pg_loss = -(log_probs_t * advantages.detach()).sum()

        entropy_bonus = torch.stack(entropy_terms).mean()
        loss = pg_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % print_every == 0:
            avg_score = sum(all_scores[-print_every:]) / len(all_scores[-print_every:])
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Score: {score} | Avg({print_every}): {avg_score:.1f} | "
                f"Baseline: {baseline:.2f} | Loss: {loss.item():.3f} | "
                f"Best score: {best_score} | Best tile: {best_tile}"
            )

    print(
        f"Training complete | Best score: {best_score} | Best tile: {best_tile}"
    )
    return policy_net, all_scores




# Simple evaluation: run greedy (argmax) or sampled episodes
def evaluate_policy(policy_net, num_episodes=50, device="cpu"):
    env = Game2048Env()
    scores = []
    best_tile = 0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done:
            state_vec = encode_state(state)
            state_tensor = torch.from_numpy(state_vec).to(device).unsqueeze(0)

            with torch.no_grad():
                logits = policy_net(state_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

            # Try actions from most to least likely until one changes the board.
            action_order = torch.argsort(probs, descending=True).tolist()
            chosen_action = None
            for action in action_order:
                board_copy = [row[:] for row in state]
                _, _, changed = env._simulate_move(board_copy, action)
                if changed:
                    chosen_action = action
                    break

            # If no action can change the board, the episode is over.
            if chosen_action is None:
                done = True
                info = {"score": env.score}
                break

            state, reward, done, info = env.step(chosen_action)

            steps += 1
            if steps > 5000:
                print("Evaluation aborting early due to step limit; policy is stuck.")
                done = True
                break

        scores.append(info["score"])
        episode_max_tile = max(max(row) for row in env.board)
        best_tile = max(best_tile, episode_max_tile)

    print(
        f"Evaluation over {num_episodes} episodes: "
        f"avg score = {sum(scores)/len(scores):.1f}, "
        f"best score = {max(scores)}, "
        f"best tile = {best_tile}"
    )
    return scores


def plot_average_score(scores, window=50, smoothing_factor=0.9, save_path=None):
    """
    Plot running average score over episodes with optional exponential smoothing.
    window: number of episodes for moving average
    smoothing_factor: higher -> smoother (None to disable)
    save_path: if provided, save the figure to this path instead of showing
    """
    if not scores:
        print("No scores to plot.")
        return

    scores_arr = np.array(scores, dtype=np.float32)
    window = max(1, min(window, len(scores_arr)))
    kernel = np.ones(window) / window
    moving_avg = np.convolve(scores_arr, kernel, mode="valid")

    if smoothing_factor is not None:
        smoothed = np.zeros_like(moving_avg)
        smoothed[0] = moving_avg[0]
        for i in range(1, len(moving_avg)):
            smoothed[i] = smoothing_factor * smoothed[i - 1] + (1 - smoothing_factor) * moving_avg[i]
        plot_series = smoothed
        label = f"{window}-ep avg (smoothed)"
    else:
        plot_series = moving_avg
        label = f"{window}-ep avg"

    episodes = np.arange(1, len(plot_series) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, plot_series, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Average Score Over Episodes")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved average score plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_evaluation_scores(scores, window=10, smoothing_factor=0.9, save_path=None):
    """
    Plot evaluation scores per episode with optional moving average and smoothing.
    window: number of episodes for moving average
    smoothing_factor: higher -> smoother (None to disable)
    save_path: if provided, save the figure to this path instead of showing
    """
    if not scores:
        print("No evaluation scores to plot.")
        return

    scores_arr = np.array(scores, dtype=np.float32)
    window = max(1, min(window, len(scores_arr)))
    kernel = np.ones(window) / window
    moving_avg = np.convolve(scores_arr, kernel, mode="valid")

    if smoothing_factor is not None:
        smoothed = np.zeros_like(moving_avg)
        smoothed[0] = moving_avg[0]
        for i in range(1, len(moving_avg)):
            smoothed[i] = smoothing_factor * smoothed[i - 1] + (1 - smoothing_factor) * moving_avg[i]
        plot_series = smoothed
        label = f"{window}-ep eval avg (smoothed)"
    else:
        plot_series = moving_avg
        label = f"{window}-ep eval avg"

    episodes = np.arange(1, len(plot_series) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, plot_series, label=label, color="orange")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Score")
    plt.title("Evaluation Scores Over Episodes")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved evaluation score plot to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    policy_net, scores = train_reinforce(
        
    )
    """
        Optimization notes:
        num_episodes: total number of full games (episodes) to train
        gamma:  discount factor, higher -> longer-term rewards matter more
        lr: learning rate for Adam optimizer, smaller -> slower learning but more stable
        print_every: frequency of logging progress
        device: "cpu" or "cuda"
        baseline_momentum: EMA factor for baseline; higher -> slower updates
        entropy_coef: weight for entropy bonus to encourage exploration, smaller -> less exploration
        max_steps_per_episode: safety cap to prevent infinite loops in stuck policies
    """

    plot_average_score(scores, window=100, smoothing_factor=0.9, save_path=None)
    eval_scores = evaluate_policy(policy_net, num_episodes=50, device=device)
    plot_evaluation_scores(eval_scores, window=10, smoothing_factor=0.9, save_path=None)
    print("Training and evaluation complete.")

# policy_pg_2048.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

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
    baseline_momentum=0.9,
    entropy_coef=0.01,
):
    env = Game2048Env()
    policy_net = PolicyNet().to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    all_scores = []
    best_score = float("-inf")
    best_tile = 0
    baseline_mean = 0.0

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        log_probs = []
        entropies = []
        rewards = []
        done = False

        while not done:
            action, log_prob, entropy = select_action(policy_net, state, device=device)
            next_state, reward, done, info = env.step(action)  # shaped reward
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            state = next_state

        # Episode finished
        score = info["score"]
        all_scores.append(score)
        best_score = max(best_score, score)
        # Track the largest tile reached this episode.
        episode_max_tile = max(max(row) for row in env.board)
        best_tile = max(best_tile, episode_max_tile)

        # Compute returns and build advantages with a running baseline
        returns = compute_returns(rewards, gamma=gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        episode_mean = returns.mean().item()
        baseline_mean = baseline_momentum * baseline_mean + (1 - baseline_momentum) * episode_mean
        advantages = returns - baseline_mean

        # Policy gradient loss per episode
        log_probs_tensor = torch.stack(log_probs)
        advantages_tensor = advantages  # already a tensor
        loss = -(log_probs_tensor * advantages_tensor).mean()
        if entropies:
            entropy_term = torch.stack(entropies).mean()
            loss -= entropy_coef * entropy_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % print_every == 0:
            avg_score = sum(all_scores[-print_every:]) / len(all_scores[-print_every:])
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Last score: {score} | Avg({print_every}): {avg_score:.1f} | "
                f"Loss: {loss.item():.3f} | Current Max score: {best_score} | "
                f"Current Max tile: {best_tile}"
            )

    if all_scores:
        final_best = max(all_scores)
        print(
            f"Training complete | Best score across all episodes: {final_best} | "
            f"Best tile across all episodes: {best_tile}"
        )

    return policy_net, all_scores


# Simple evaluation: run greedy (argmax) or sampled episodes
def evaluate_policy(policy_net, num_episodes=50, device="cpu"):
    env = Game2048Env()
    scores = []

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

    print(
        f"Evaluation over {num_episodes} episodes: "
        f"avg score = {sum(scores)/len(scores):.1f}, "
    )
    return scores


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    policy_net, scores = train_reinforce(
        num_episodes=2000,
        gamma=0.99,
        lr=1e-3,
        print_every=50,
        device=device
    )

    evaluate_policy(policy_net, num_episodes=50, device=device)
    print("Training and evaluation complete.")

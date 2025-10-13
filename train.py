import random

import numpy as np
import torch

import environment
from agent import Agent
from environment import Action, TetrisEnv
from utils import plot_training_progress

# Config
NUM_EPISODES = 500


def set_global_seeds(seed: int = 42) -> None:
    """
    Set the global random seeds for reproducibility.

    Parameters
    ----------
    seed : int, optional
        The random seed to use. Default: ``42``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train() -> None:
    """Main training function for the reinforcement learning agent."""
    device = torch.device("cuda", torch.cuda.current_device())
    print(f"Training on device: {device}")

    scores, mean_scores = [], []
    total_score = 0
    env = TetrisEnv()
    agent = Agent(device, len(env.get_state()), len(Action))

    for episode in range(NUM_EPISODES):
        env.reset()
        state = env.get_state()
        score = 0
        done = False

        while not done:
            action = agent.get_action(state)
            reward, score, done = env.step(action)
            next_state = env.get_state()
            agent.store_transition(state, action, next_state, reward, done)
            state = next_state
            # TODO: optimization of the model
            # TODO: syncing of the target network

        scores.append(score)
        total_score += score
        mean_score = total_score / (episode + 1)
        mean_scores.append(mean_score)

        print(
            f"Episode {episode}/{NUM_EPISODES} - Score: {score} - Mean Score: {mean_score}"
        )
        plot_training_progress(scores, mean_scores)

    print("Training complete")
    plot_training_progress(
        scores,
        mean_scores,
        save_and_close=True,
        file_name=f"training_progess_dqn_{NUM_EPISODES}_episodes.pdf",
    )
    agent.save_model(file_name=f"dqn_{NUM_EPISODES}_episodes.pt")


if __name__ == "__main__":
    assert (
        torch.cuda.is_available()
    ), "CUDA not available. Please check your PyTorch installation."

    set_global_seeds()
    train()

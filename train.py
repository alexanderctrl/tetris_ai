import argparse
import cProfile
import io
import pstats
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from agent import Agent
from environment import Action, TetrisEnv
from utils import plot_training_progress

# Config
NUM_EPISODES = 500


@contextmanager
def profiling():
    """Context manager for profiling a code block."""
    pr = cProfile.Profile()
    if args.profile:
        pr.enable()
    yield pr
    if args.profile:
        pr.disable()


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

    env = TetrisEnv(args.headless)
    agent = Agent(device, env.get_state()[0].shape[0], len(Action))
    scores, mean_scores = [], []
    discounted_returns, mean_discounted_returns = [], []
    total_score, total_discounted_return = 0, 0
    discount_factor = agent.get_gamma()

    for episode in range(NUM_EPISODES):
        env.reset()
        state, valid_actions = env.get_state()
        step = 0
        score = 0
        discounted_return = 0
        done = False

        while not done:
            action = agent.get_action(state, valid_actions)
            reward, score, done = env.step(action)
            discounted_return += discount_factor**step * reward
            next_state, valid_actions = env.get_state()

            agent.store_transition(state, action, next_state, reward, done)
            state = next_state

            agent.optimize_model()
            agent.soft_update_target_network()
            step += 1

        scores.append(score)
        discounted_returns.append(discounted_return)
        total_score += score
        mean_score = total_score / (episode + 1)
        total_discounted_return += discounted_return
        mean_discounted_return = total_discounted_return / (episode + 1)
        mean_scores.append(mean_score)
        mean_discounted_returns.append(mean_discounted_return)

        print(f"--------------- Episode {episode+1}/{NUM_EPISODES} ---------------")
        print(f"Score: {score} - Mean Score: {mean_score:.2f}")
        print(
            f"Discounted Return: {discounted_return:.2f} - Mean Discounted Return: {mean_discounted_return:.2f}"
        )
        plot_training_progress(
            scores, mean_scores, discounted_returns, mean_discounted_returns
        )

    print("Training complete")
    plot_training_progress(
        scores,
        mean_scores,
        discounted_returns,
        mean_discounted_returns,
        save_and_close=True,
        file_name=f"training_progess_dqn_{NUM_EPISODES}_episodes.pdf",
    )
    agent.save_model(file_name=f"dqn_{NUM_EPISODES}_episodes.pt")


if __name__ == "__main__":
    assert (
        torch.cuda.is_available()
    ), "CUDA not available. Please check your PyTorch installation."

    parser = argparse.ArgumentParser(description="Train a DQN agent to play Tetris.")
    parser.add_argument(
        "--headless", help="run the environment in headless mode", action="store_true"
    )
    parser.add_argument(
        "--profile", help="enable profiling for the training loop", action="store_true"
    )
    args = parser.parse_args()

    start_time = time.perf_counter()

    with profiling() as pr:
        set_global_seeds()
        train()

    elapsed_time = time.perf_counter() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    if args.profile:
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(50)
        print(s.getvalue())

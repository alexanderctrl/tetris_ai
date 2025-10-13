import math
import random
from collections import deque, namedtuple

import numpy as np
import torch

from dqn import DQN
from environment import Action, Tetris

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayMemory(object):
    """
    Cyclic buffer that stores recently observed transitions.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions that can be stored.
    """

    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: int,
        done: bool,
    ) -> None:
        """
        Store a new transition. If the maximum capacity is reached, replace the
        oldest transition by the new one.

        Parameters
        ----------
        state : np.ndarray
            State of the environment.
        action : int
            Action performed by the agent.
        next_state : np.ndarray
            Resulting state of the environment after performing the action.
        reward : int
            Reward received after performing the action.
        done: bool
            Flag that indicates a terminal state.
        """
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size: int) -> list[Transition]:
        """
        Return a random sample of unique transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions in the sample.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class Agent:
    """
    Agent that interacts with the environment using a Deep Q-Network (DQN).

    This class handles action selection, memory storage, and model optimization
    for the reinforcement learning task.

    Parameters
    ----------
    device : torch.device
        The device on which the agent's neural networks and tensors will be allocated.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

        self.game = Tetris()
        self.memory = ReplayMemory(100_000)
        self.policy_net = DQN(len(self.game.get_state()), len(Action)).to(self.device)
        self.target_net = DQN(len(self.game.get_state()), len(Action)).to(self.device)

        self.num_steps = 0
        self.eps_start = 1
        self.eps_end = 0.1
        self.eps_decay = 10_000

        self.batch_size = 128
        # TODO: finish initialization

    def get_action(self, state: np.ndarray) -> Action:
        """
        Get the next action based on an epsilon greedy policy.

        Parameters
        ----------
        state : np.ndarray
            State of the environment.
        """
        eps_sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start + self.eps_end) * math.exp(
            -1 * self.num_steps / self.eps_decay
        )
        self.num_steps += 1
        if eps_sample > eps_threshold:
            return random.choice(list(Action))  # TODO: change to query DQN
        else:
            return random.choice(list(Action))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: int,
        done: bool,
    ) -> None:
        pass  # TODO

    def optimize_model(self) -> None:
        pass  # TODO

    def sync_target_network(self) -> None:
        pass  # TODO

    def save_model(self, file_name: str = "model.pt") -> None:
        """
        Save the model parameters of the target network to a file.

        Parameters
        ----------
        file_name : str, optional
            Name of the file to save the model parameters to. Default: ``"model.pt"``.
        """
        self.target_net.save(file_name)

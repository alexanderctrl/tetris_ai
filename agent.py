import math
import random
from collections import deque, namedtuple

import numpy as np
import torch

from dqn import DQN
from dqn_trainer import DQN_Trainer
from environment import Action

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
        action: np.int64,
        next_state: np.ndarray,
        reward: np.float32,
        done: np.float32,
    ) -> None:
        """
        Store a new transition. If the maximum capacity is reached, replace the
        oldest transition by the new one.

        Parameters
        ----------
        state : np.ndarray
            State of the environment.
        action : np.int64
            Action performed by the agent.
        next_state : np.ndarray
            Resulting state of the environment after performing the action.
        reward : np.float32
            Reward received after performing the action.
        done: np.float32
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

    This class handles action selection, memory storage, and model synchronization
    for the reinforcement learning task.

    Parameters
    ----------
    device : torch.device
        The device on which the agent's neural networks and tensors will be allocated.
    num_channels : int
        Number of channels representing the environment's state. Each channel is a 20x10 board that encodes a specific characteristic of the game.
    num_actions : int
        Number of possible actions.
    """

    def __init__(
        self, device: torch.device, num_channels: int, num_actions: int
    ) -> None:
        self.device = device
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.01

        self.policy_net = DQN(num_channels, num_actions).to(self.device)
        self.target_net = DQN(num_channels, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayMemory(100_000)
        self.trainer = DQN_Trainer(self.policy_net, self.target_net, self.gamma)

        self.num_steps = 0
        self.eps_start = 1
        self.eps_end = 0.025
        self.eps_decay = 10_000

    def get_action(self, state: np.ndarray, valid_actions: list[Action]) -> Action:
        """
        Get the next action based on an epsilon greedy policy, restricted to valid actions.

        Parameters
        ----------
        state : np.ndarray
            State of the environment.
        valid_actions : list[Action]
            List of valid actions available in the current state.
        """
        eps_sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1 * self.num_steps / self.eps_decay
        )
        self.num_steps += 1
        if eps_sample > eps_threshold:
            with torch.no_grad():
                q_value_preds = (
                    self.policy_net(
                        torch.tensor(state, dtype=torch.float32).to(self.device)
                    )
                    .cpu()
                    .numpy()
                    .squeeze()
                )

                return max(valid_actions, key=lambda a: q_value_preds[a.value])
        else:
            return random.choice(valid_actions)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.memory.push(
            state, np.int64(action), next_state, np.float32(reward), np.float32(done)
        )

    def optimize_model(self) -> None:
        """
        Perform a single optimization step for the agent's policy network.

        This method performs the following steps:
        1. Samples a batch of transitions from the agent's replay memory.
        2. Converts the batch elements to tensors and moves them to the correct device.
        3. Passes the batch to the trainer, which computes the temporal difference
        error and performs the gradient descent step on the policy network.

        If the replay memory contains fewer transitions than the batch size,
        no optimization step is performed.
        """
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*transitions)

        states = np.stack(states)
        next_states = np.stack(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = (
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.trainer.optimize_model(states, actions, next_states, rewards, dones)

    def soft_update_target_network(self) -> None:
        """
        Perform a soft update of the target network parameters toward the policy
        network using Polyak averaging.
        """
        policy_net_state_dict = self.policy_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (
                self.tau * policy_net_state_dict[key]
                + (1 - self.tau) * target_net_state_dict[key]
            )
        self.target_net.load_state_dict(target_net_state_dict)

    def save_model(self, file_name: str = "model.pt") -> None:
        """
        Save the model parameters of the target network to a file.

        Parameters
        ----------
        file_name : str, optional
            Name of the file to save the model parameters to. Default: ``"model.pt"``.
        """
        self.target_net.save(file_name)

    def get_gamma(self) -> float:
        return self.gamma

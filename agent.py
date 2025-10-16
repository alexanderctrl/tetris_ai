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

    This class handles action selection, memory storage, and model synchronization
    for the reinforcement learning task.

    Parameters
    ----------
    device : torch.device
        The device on which the agent's neural networks and tensors will be allocated.
    dim_state : int
        Dimensionality of the state space (number of input features).
    dim_action : int
        Dimensionality of the action space (number of possible actions).
    """

    def __init__(self, device: torch.device, dim_state: int, dim_action: int) -> None:
        self.device = device
        self.batch_size = 128
        self.gamma = 0.9
        self.tau = 0.01

        self.policy_net = DQN(dim_state, dim_action).to(self.device)
        self.target_net = DQN(dim_state, dim_action).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayMemory(100_000)
        self.trainer = DQN_Trainer(self.policy_net, self.target_net, self.gamma)

        self.num_steps = 0
        self.eps_start = 1
        self.eps_end = 0.1
        self.eps_decay = 10_000

    def get_action(self, state: np.ndarray) -> Action:
        """
        Get the next action based on an epsilon greedy policy.

        Parameters
        ----------
        state : np.ndarray
            State of the environment.
        """
        eps_sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1 * self.num_steps / self.eps_decay
        )
        self.num_steps += 1
        if eps_sample > eps_threshold:
            with torch.no_grad():
                q_value_preds = self.policy_net(
                    torch.tensor(state, dtype=torch.float32).to(self.device)
                )
                return Action(torch.argmax(q_value_preds).item())
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
        self.memory.push(state, action, next_state, reward, done)

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

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
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

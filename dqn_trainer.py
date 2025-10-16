import torch
import torch.nn as nn
import torch.optim as optim

from dqn import DQN


class DQN_Trainer:
    """
    Trainer for optimizing the Deep Q-Network (DQN) using batches of experience.

    This class encapsulates the optimization logic for the DQN. It is responsible for:
    - Computing the temporal difference (TD) error between the policy network's
    predicted Q-values and the TD targets.
    - Performing gradient descent updates on the policy network using the loss
    computed based on the TD error.

    Parameters
    ----------
    policy_net : DQN
        The policy network to optimize.
    target_net : DQN
        The target network used to compute stable TD targets.
    gamma : float
        Discount factor for computing the TD targets.
    """

    def __init__(self, policy_net: DQN, target_net: DQN, gamma: float) -> None:
        self.policy_net = policy_net
        self.target_net = target_net
        self.gamma = gamma

        self.lr = 0.001
        self.optimizer = optim.AdamW(policy_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def optimize_model(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """
        Compute the temporal difference (TD) error between the Q-values predicted
        by the policy network and the TD targets derived from the target network.
        The TD target for each transition is given by:
            target = reward + gamma * max_a' Q_target(next_state, a') * (1 - done)

        The loss measure is then computed based on the TD error and used to
        perform a gradient descent step on the policy network parameters.

        Parameters
        ----------
        states : torch.Tensor
            Batch of current states, of shape (batch_size, dim_state).
        actions : torch.Tensor
            Batch of actions taken, of shape (batch_size, 1).
        next_states : torch.Tensor
            Batch of next states, of shape (batch_size, dim_state).
        rewards : torch.Tensor
            Batch of observed rewards, of shape (batch_size, 1).
        dones: torch.Tensor
            Batch of boolean flags indicating terminal states, of shape (batch_size, 1).
        """
        predicted_q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1).values.unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.criterion(predicted_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

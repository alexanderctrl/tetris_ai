from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) that approximates the Q-function for reinforcement learning.
    It maps input states to action-value estimates.

    Parameters
    ----------
    num_channels : int
        Number of channels representing the environment's state. Each channel is a 20x10 board that encodes a specific characteristic of the game.
    num_actions : int
        Number of possible actions.
    """

    def __init__(self, num_channels: int, num_actions: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 10 * 5, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor representing the current batch of states of shape (batch_size, num_channels, height, width).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save(self, file_name: str) -> None:
        """
        Save the model parameters to a file.

        The method creates a "models" directory if it does not exist and saves
        the model's state dictionary to the specified file.

        Parameters
        ----------
        file_name : str
            Name of the file to save the model parameters to.
        """
        model_folder = Path("models")
        model_folder.mkdir(parents=True, exist_ok=True)

        file_path = model_folder / file_name
        torch.save(self.state_dict(), file_path)

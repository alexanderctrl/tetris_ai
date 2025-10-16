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
    dim_state : int
        Dimensionality of the state space (number of input features).
    dim_action : int
        Dimensionality of the action space (number of possible actions).
    """

    def __init__(self, dim_state: int, dim_action: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(dim_state, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, dim_action)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor representing the current batch of states of shape (batch_size, dim_state).
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

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

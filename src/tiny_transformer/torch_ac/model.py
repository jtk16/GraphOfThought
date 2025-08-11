import torch.nn as nn

class ACModel(nn.Module):
    """Base class for actor-critic models."""

    def __init__(self):
        super().__init__()

    def forward(self, obs):
        """
        Forward pass of the model.

        :param obs: observation
        :return: a tuple (action_distribution, value)
        """
        raise NotImplementedError

    def act(self, obs):
        """
        Returns the action to take and its value.

        :param obs: observation
        :return: a tuple (action, value)
        """
        dist, value = self(obs)
        action = dist.sample()

        return action

    def get_value(self, obs):
        """
        Returns the value of the given observation.

        :param obs: observation
        :return: value
        """
        dist, value = self(obs)

        return value

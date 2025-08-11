import torch
import torch.nn as nn
from torch.autograd import Variable

from ..utils import DictList, ParallelEnv

class BaseAlgo(object):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 reshape_reward=None):
        self.envs = envs
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.reshape_reward = reshape_reward
        self.num_procs = len(envs) # Add this line

        self.optimizer = None

    def collect_experiences(self):
        """Collects experiences for a given number of frames."""
        shape = (self.num_frames_per_proc, len(self.envs))
        
        obs = self.envs.reset()
        
        obss = [None]*(self.num_frames_per_proc)
        actions = [None]*(self.num_frames_per_proc)
        values = [None]*(self.num_frames_per_proc)
        rewards = [None]*(self.num_frames_per_proc)
        terminals = [None]*(self.num_frames_per_proc)
        masks = [None]*(self.num_frames_per_proc)
        
        for i in range(self.num_frames_per_proc):
            # Do one step in the environment
            action = self.acmodel.act(obs)
            new_obs, reward, terminal, _ = self.envs.step(action)

            # Update experiences
            obss[i] = obs
            actions[i] = action
            values[i] = self.acmodel.get_value(obs)
            rewards[i] = reward
            terminals[i] = terminal
            masks[i] = 1 - terminal.float()

            obs = new_obs

        # Add advantage and return to experiences
        exps = DictList({
            "obs": obss,
            "action": actions,
            "value": values,
            "reward": rewards,
            "terminal": terminals,
            "mask": masks
        })

        return exps

    def update_parameters(self, exps):
        """Updates the model's parameters."""
        raise NotImplementedError

    def get_batches_starting_indexes(self):
        """Returns the indexes where to start the batches."""
        indexes = self.num_frames_per_proc * self.num_procs
        indexes = torch.arange(0, indexes, self.recurrence).long()
        indexes = indexes[torch.randperm(len(indexes))]

        return indexes

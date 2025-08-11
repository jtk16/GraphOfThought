import torch
import torch.nn.functional as F

from .base import BaseAlgo

class A2CAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95, 
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4, 
                 optim_alpha=0.99, optim_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256,
                 reshape_reward=None):
        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(
            self.acmodel.parameters(), lr, eps=optim_eps)

    def update_parameters(self, exps):
        # Collect experiences
        # exps = self.collect_experiences()

        # Initialize log values
        log_entropies = []
        log_values = []
        log_policy_losses = []
        log_value_losses = []
        log_grad_norms = []

        for _ in range(self.epochs):
            # Initialize batch values
            batch_indexes = self.get_batches_starting_indexes()
            batch_size = self.batch_size if self.batch_size % self.recurrence == 0 else \
                self.batch_size - (self.batch_size % self.recurrence)

            for i in range(len(batch_indexes) // batch_size):
                batch_index = batch_indexes[i * batch_size:(i + 1) * batch_size]
                exps_batch = exps[batch_index]

                # Compute loss
                dist, value = self.acmodel(exps_batch.obs)

                entropy = dist.entropy().mean()
                log_probs = dist.log_prob(exps_batch.action)

                policy_loss = -(exps_batch.advantage * log_probs).mean()
                value_loss = (value - exps_batch.returnn).pow(2).mean()
                
                loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                # Update actor-critic
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values
                log_entropies.append(entropy.item())
                log_values.append(value.mean().item())
                log_policy_losses.append(policy_loss.item())
                log_value_losses.append(value_loss.item())
                log_grad_norms.append(grad_norm)

        # Log some values
        logs = {
            "entropy": sum(log_entropies) / len(log_entropies),
            "value": sum(log_values) / len(log_values),
            "policy_loss": sum(log_policy_losses) / len(log_policy_losses),
            "value_loss": sum(log_value_losses) / len(log_value_losses),
            "grad_norm": sum(log_grad_norms) / len(log_grad_norms)
        }

        return logs
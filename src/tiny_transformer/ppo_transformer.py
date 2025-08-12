import torch
import torch.nn as nn
from .torch_ac.algos import PPOAlgo
from .torch_ac.model import ACModel

class PPOTransformer(ACModel):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length):
        super(PPOTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Use batch_first so tensors are (batch, seq, feature)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers)
        self.actor = nn.Linear(d_model, vocab_size)
        self.critic = nn.Linear(d_model, 1)

    def forward(self, obs):
        x = self.embedding(obs.text)
        x = self.transformer_encoder(x)
        
        # Take the output corresponding to the last token for actor and critic
        # Assuming batch_first=True in transformer_encoder, x shape is (batch_size, seq_len, d_model)
        last_token_output = x[:, -1, :] # (batch_size, d_model)
        
        dist = torch.distributions.Categorical(logits=self.actor(last_token_output))
        value = self.critic(last_token_output).squeeze(1)

        return dist, value

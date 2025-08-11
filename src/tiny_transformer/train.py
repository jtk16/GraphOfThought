import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .math_env import MathEnv
from .torch_ac.utils.dictlist import DictList
from .utils import tokenize_string, detokenize_tensor

class MathDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.samples = []
        for _ in range(num_samples):
            a = torch.randint(0, 5, (1,)).item()
            b = torch.randint(0, 5, (1,)).item()
            op = torch.randint(0, 4, (1,)).item()
            if op == 0:
                question = f"{a}+{b}"
                answer = str(a + b)
            elif op == 1:
                question = f"{a}-{b}"
                answer = str(a - b)
            elif op == 2:
                question = f"{a}*{b}"
                answer = str(a * b)
            else:
                question = f"{a}/{b}"
                # Use integer division and return 'inf' for invalid cases
                answer = str(a // b) if b != 0 and a % b == 0 else "inf"
            self.samples.append((question, answer))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train_standard_transformer(model, dataset, vocab, epochs=10, max_seq_length=20):
    dataloader = DataLoader(dataset, batch_size=32)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    for epoch in range(epochs):
        for batch in dataloader:
            questions, answers = batch
            if hasattr(model, 'base_transformer'): # This is a KVTGIntegratedTransformer
                device = model.base_transformer.embedding.weight.device
            else: # This is a StandardTransformer
                device = model.embedding.weight.device
            src = torch.tensor([tokenize_string(q, vocab, max_seq_length) for q in questions]).to(device)
            tgt = torch.tensor([tokenize_string(a, vocab, max_seq_length) for a in answers]).to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, len(vocab)), tgt.view(-1))
            loss.backward()
            optimizer.step()

def train_kvtg_transformer(model, dataset, vocab, epochs=10, max_seq_length=20):
    dataloader = DataLoader(dataset, batch_size=32)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            questions, answers = batch
            if hasattr(model, 'base_transformer'): # This is a KVTGIntegratedTransformer
                device = model.base_transformer.embedding.weight.device
            else: # This is a StandardTransformer
                device = model.embedding.weight.device
            src = torch.tensor([tokenize_string(q, vocab, max_seq_length) for q in questions]).to(device)
            tgt = torch.tensor([tokenize_string(a, vocab, max_seq_length) for a in answers]).to(device)
            
            optimizer.zero_grad()
            
            # For KVTG, we'll pass use_kvtg=True and a dummy problem_id
            # In a real scenario, problem_id would be unique for each problem
            output = model(src, tgt, use_kvtg=True, problem_id=f"problem_{i}")
            loss = criterion(output.view(-1, len(vocab)), tgt.view(-1))
            loss.backward()
            optimizer.step()

def train_seal_integrated_transformer(model, dataset, vocab, rev_vocab, epochs=10, max_seq_length=20):
    dataloader = DataLoader(dataset, batch_size=32)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            questions, answers = batch
            device = model.kvtg_integrated_model.base_transformer.embedding.weight.device
            src = torch.tensor([tokenize_string(q, vocab, max_seq_length) for q in questions]).to(device)
            tgt = torch.tensor([tokenize_string(a, vocab, max_seq_length) for a in answers]).to(device)
            
            optimizer.zero_grad()
            
            # Check if the answer is correct to trigger SEAL adaptation
            is_successful = False
            # Simplified check: if the first character of the predicted answer matches the true answer
            # In a real scenario, this would involve a more robust evaluation of the generated thought graph
            output = model(src, tgt, use_kvtg=True, problem_id=f"problem_{i}")
            
            if output is not None:
                predicted_answer_str = detokenize_tensor(output, rev_vocab)
                try:
                    if eval(questions[0]) == int(predicted_answer_str):
                        is_successful = True
                except (SyntaxError, ValueError, ZeroDivisionError):
                    pass
            
            # Pass is_successful to the model's forward pass to trigger SEAL adaptation
            output = model(src, tgt, use_kvtg=True, problem_id=f"problem_{i}", is_successful=is_successful)
            
            loss = criterion(output.view(-1, len(vocab)), tgt.view(-1))
            loss.backward()
            optimizer.step()

def train_ppo_transformer(model, dataset, vocab, rev_vocab, epochs=10, gamma=0.99):
    """Train PPO transformer on the MathEnv environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    env = MathEnv(vocab, rev_vocab)

    for _ in range(epochs):
        obs, _ = env.reset()
        done = False

        log_probs = []
        values = []
        rewards = []

        while not done:
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
            dist, value = model(DictList({"text": obs_tensor}))
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            values.append(value)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(torch.tensor([reward], device=device))

        # Compute returns
        returns = []
        G = torch.tensor([0.0], device=device)
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.cat(returns)
        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze(-1)

        advantages = returns - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = (returns - values).pow(2).mean()
        entropy = - (log_probs.exp() * log_probs).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
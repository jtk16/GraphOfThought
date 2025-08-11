import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .standard_transformer import StandardTransformer
from .torch_ac.algos import PPOAlgo
from .kvtg_integration import KVTGIntegratedTransformer
from .seal_integration import SEALIntegratedTransformer
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
                answer = str(a / b) if b != 0 else "inf"
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

def train_ppo_transformer(model, dataset, vocab, rev_vocab, epochs=10):
    # PPO training is more involved and requires an environment.
    # This is a placeholder for a more complete implementation.
    print("PPO training is not yet implemented.")
    
    # Create a dummy environment for PPO
    env = MathEnv(vocab, rev_vocab)
    
    # PPOAlgo expects a model that inherits from ACModel
    # Our PPOTransformer inherits from ACModel
    
    # Create a PPO agent
    algo = PPOAlgo([env], model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_frames_per_proc=1) # Added num_frames_per_proc
    
    num_frames = 1000 # Number of frames to train for
    
    # Collect experiences and update model
    obs = env.reset()
    for i in range(num_frames):
        # Convert numpy array observation to DictList
        obs_dict = DictList({"text": torch.tensor(obs[0]).unsqueeze(0)})
        
        # Get action from model
        dist, value = model(obs_dict)
        action = dist.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        # Update algo
        # This is a very simplified update. A real PPO loop is more complex.
        # It involves storing experiences, computing advantages, and updating the model.
        # For now, we'll just do a dummy update.
        
        # Create dummy experiences for the algo
        exps = DictList({
            "obs": obs_dict,
            "action": action.unsqueeze(0),
            "value": value.unsqueeze(0),
            "reward": torch.tensor([reward]).unsqueeze(0),
            "mask": torch.tensor([1-done]).unsqueeze(0),
            "log_prob": dist.log_prob(action).unsqueeze(0),
            "advantage": torch.tensor([0.0]).unsqueeze(0), # Dummy advantage
            "returnn": torch.tensor([0.0]).unsqueeze(0) # Dummy return
        })
        
        # Update the algorithm
        algo.update_parameters(exps)
        
        if done:
            obs = env.reset()
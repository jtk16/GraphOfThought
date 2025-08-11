import torch
from .math_env import MathEnv
from .torch_ac.utils.dictlist import DictList
from .utils import tokenize_string, detokenize_tensor

def evaluate_standard_transformer(model, dataset, vocab, rev_vocab, max_seq_length=20):
    correct = 0
    for question, answer in dataset:
        if hasattr(model, 'base_transformer'): # This is a KVTGIntegratedTransformer
            device = model.base_transformer.embedding.weight.device
        else: # This is a StandardTransformer
            device = model.embedding.weight.device
        src = torch.tensor([tokenize_string(question, vocab, max_seq_length)]).to(device)
        
        # In evaluation, we typically generate token by token
        # For simplicity, we'll just take the output of a single forward pass
        output = model(src, src) # Dummy tgt for now
        
        predicted_answer_str = detokenize_tensor(output, rev_vocab)
        
        try:
            # Evaluate the generated answer and compare with the true answer
            if eval(question) == int(predicted_answer_str):
                correct += 1
        except (SyntaxError, ValueError, ZeroDivisionError):
            if predicted_answer_str == "inf":
                correct += 1
    return correct / len(dataset)

def evaluate_kvtg_transformer(model, dataset, vocab, rev_vocab, max_seq_length=20):
    correct = 0
    for i, (question, answer) in enumerate(dataset):
        if hasattr(model, 'base_transformer'): # This is a KVTGIntegratedTransformer
            device = model.base_transformer.embedding.weight.device
        else: # This is a StandardTransformer
            device = model.embedding.weight.device
        src = torch.tensor([tokenize_string(question, vocab, max_seq_length)]).to(device)
        
        # For KVTG, we'll pass use_kvtg=True and a dummy problem_id
        output = model(src, src, use_kvtg=True, problem_id=f"problem_{i}") # Simplified generation
        
        # Check if output is None (e.g., if KVTG exploration failed)
        if output is None:
            continue

        predicted_answer_str = detokenize_tensor(output, rev_vocab)
        
        try:
            if eval(question) == int(predicted_answer_str):
                correct += 1
        except (SyntaxError, ValueError, ZeroDivisionError):
            if predicted_answer_str == "inf":
                correct += 1
        
    return correct / len(dataset)

def evaluate_seal_integrated_transformer(model, dataset, vocab, rev_vocab, max_seq_length=20):
    correct = 0
    for i, (question, answer) in enumerate(dataset):
        # For SEAL, we need to get the device from the nested KVTG model
        device = model.kvtg_integrated_model.base_transformer.embedding.weight.device
        src = torch.tensor([tokenize_string(question, vocab, max_seq_length)]).to(device)
        
        # For SEAL, we'll pass use_kvtg=True and a dummy problem_id
        output = model(src, src, use_kvtg=True, problem_id=f"problem_{i}") # Simplified generation
        
        # Check if output is None (e.g., if KVTG exploration failed)
        if output is None:
            continue

        predicted_answer_str = detokenize_tensor(output, rev_vocab)
        
        try:
            if eval(question) == int(predicted_answer_str):
                correct += 1
        except (SyntaxError, ValueError, ZeroDivisionError):
            if predicted_answer_str == "inf":
                correct += 1
        
    return correct / len(dataset)

def evaluate_ppo_transformer(model, dataset, vocab, rev_vocab, episodes=100):
    """Evaluate PPO transformer by running episodes in the environment."""
    env = MathEnv(vocab, rev_vocab)
    success = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        final_reward = 0

        while not done:
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(device)
            dist, _ = model(DictList({"text": obs_tensor}))
            action = torch.argmax(dist.probs, dim=-1)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            final_reward = reward

        if final_reward > 0:
            success += 1

    return success / episodes

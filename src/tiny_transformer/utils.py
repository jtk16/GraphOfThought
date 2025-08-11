import torch

def tokenize_string(s, vocab, max_len):
    # Convert string to a list of token IDs
    token_ids = [vocab[char] for char in s if char in vocab]
    # Pad or truncate to max_len
    if len(token_ids) < max_len:
        token_ids.extend([vocab['<pad>']] * (max_len - len(token_ids))) # Use vocab['<pad>'] for padding
    else:
        token_ids = token_ids[:max_len]
    return token_ids

def detokenize_tensor(tensor, rev_vocab):
    # Convert a tensor of token IDs back to a string
    # Assuming tensor is (batch_size, seq_len, vocab_size) or (seq_len, vocab_size)
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0) # Remove batch dimension if present
    
    predicted_ids = torch.argmax(tensor, dim=-1).flatten().tolist()
    # Filter out padding tokens and convert to characters
    predicted_chars = [rev_vocab[idx] for idx in predicted_ids if rev_vocab[idx] != '<pad>']
    return "".join(predicted_chars)

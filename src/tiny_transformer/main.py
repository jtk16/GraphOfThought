import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .train import MathDataset, train_standard_transformer, train_ppo_transformer, train_kvtg_transformer, train_seal_integrated_transformer
from .evaluate import evaluate_standard_transformer, evaluate_ppo_transformer, evaluate_kvtg_transformer, evaluate_seal_integrated_transformer
from .standard_transformer import StandardTransformer
from .ppo_transformer import PPOTransformer
from .kvtg_integration import KVTGIntegratedTransformer
from .seal_integration import SEALIntegratedTransformer

# Define vocabulary
all_chars = [str(i) for i in range(10)] + ['+', '-', '*', '/', 'inf', '.', '-']
all_chars = sorted(list(set(all_chars))) # Ensure uniqueness and consistent order

vocab = {char: i for i, char in enumerate(all_chars)}

# PADDING_TOKEN will be the next available index
PADDING_TOKEN = len(vocab)
vocab['<pad>'] = PADDING_TOKEN

# Create reverse vocabulary for decoding
rev_vocab = {v: k for k, v in vocab.items()}

# Model hyperparameters
VOCAB_SIZE = len(vocab)
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 512
MAX_SEQ_LENGTH = 20

def main():
    # Create dataset
    dataset = MathDataset()

    print(f"Vocab: {vocab}")
    print(f"VOCAB_SIZE: {VOCAB_SIZE}")

    # Create and train standard transformer
    standard_model = StandardTransformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH)
    train_standard_transformer(standard_model, dataset, vocab, epochs=10, max_seq_length=MAX_SEQ_LENGTH)
    standard_accuracy = evaluate_standard_transformer(standard_model, dataset, vocab, rev_vocab, max_seq_length=MAX_SEQ_LENGTH)
    print(f"Standard Transformer Accuracy: {standard_accuracy}")

    # Create and train KVTG-integrated transformer
    kvtg_model = KVTGIntegratedTransformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH, vocab, rev_vocab)
    train_kvtg_transformer(kvtg_model, dataset, vocab, epochs=10, max_seq_length=MAX_SEQ_LENGTH)
    kvtg_accuracy = evaluate_kvtg_transformer(kvtg_model, dataset, vocab, rev_vocab, max_seq_length=MAX_SEQ_LENGTH)
    print(f"KVTG Integrated Transformer Accuracy: {kvtg_accuracy}")

    # Create and train SEAL-integrated transformer
    seal_kvtg_model = SEALIntegratedTransformer(kvtg_model, vocab, rev_vocab)
    train_seal_integrated_transformer(seal_kvtg_model, dataset, vocab, rev_vocab, epochs=10, max_seq_length=MAX_SEQ_LENGTH)
    seal_kvtg_accuracy = evaluate_seal_integrated_transformer(seal_kvtg_model, dataset, vocab, rev_vocab, max_seq_length=MAX_SEQ_LENGTH)
    print(f"SEAL KVTG Integrated Transformer Accuracy: {seal_kvtg_accuracy}")

    # Create and train PPO transformer
    # ppo_model = PPOTransformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH)
    # train_ppo_transformer(ppo_model, dataset, vocab, rev_vocab, epochs=10)
    # ppo_accuracy = evaluate_ppo_transformer(ppo_model, dataset, vocab, rev_vocab)
    # print(f"PPO Transformer Accuracy: {ppo_accuracy}")

if __name__ == "__main__":
    main()

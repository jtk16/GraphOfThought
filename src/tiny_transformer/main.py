from .train import MathDataset, train_standard_transformer, train_ppo_transformer
from .evaluate import evaluate_standard_transformer, evaluate_ppo_transformer
from .standard_transformer import StandardTransformer
from .ppo_transformer import PPOTransformer

# Define vocabulary
vocab = {str(i): i for i in range(10)} 
vocab.update({op: i + 10 for i, op in enumerate(['+', '-', '*', '/'])})
vocab.update({'inf': 14})

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

    # Create and train standard transformer
    standard_model = StandardTransformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH)
    train_standard_transformer(standard_model, dataset, vocab)
    standard_accuracy = evaluate_standard_transformer(standard_model, dataset, vocab)
    print(f"Standard Transformer Accuracy: {standard_accuracy}")

    # Create and train PPO transformer
    ppo_model = PPOTransformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH)
    train_ppo_transformer(ppo_model, dataset, vocab)
    ppo_accuracy = evaluate_ppo_transformer(ppo_model, dataset, vocab)
    print(f"PPO Transformer Accuracy: {ppo_accuracy}")

if __name__ == "__main__":
    main()

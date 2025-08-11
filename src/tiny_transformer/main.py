"""Entry point for training and evaluating tiny transformers.

This module is normally executed as ``python -m tiny_transformer.main``.  On
Windows it is convenient to invoke it directly via ``python
src/tiny_transformer/main.py``.  The latter form does not provide a package
context, causing relative imports to fail.  To support both invocation styles we
attempt relative imports first and fall back to adjusting ``sys.path`` for
direct execution.
"""

try:  # pragma: no cover - exercised via direct script execution
    from .train import (
        MathDataset,
        train_standard_transformer,
        train_ppo_transformer,
        train_kvtg_transformer,
        train_seal_integrated_transformer,
    )
    from .evaluate import (
        evaluate_standard_transformer,
        evaluate_ppo_transformer,
        evaluate_kvtg_transformer,
        evaluate_seal_integrated_transformer,
    )
    from .standard_transformer import StandardTransformer
    from .ppo_transformer import PPOTransformer
    from .kvtg_integration import KVTGIntegratedTransformer
    from .seal_integration import SEALIntegratedTransformer
except ImportError:  # pragma: no cover - import path adjustment for script use
    import os
    import sys

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(base_dir)  # allow ``import tiny_transformer.*``
    sys.path.append(os.path.dirname(base_dir))  # allow ``import src.*``

    from tiny_transformer.train import (
        MathDataset,
        train_standard_transformer,
        train_ppo_transformer,
        train_kvtg_transformer,
        train_seal_integrated_transformer,
    )
    from tiny_transformer.evaluate import (
        evaluate_standard_transformer,
        evaluate_ppo_transformer,
        evaluate_kvtg_transformer,
        evaluate_seal_integrated_transformer,
    )
    from tiny_transformer.standard_transformer import StandardTransformer
    from tiny_transformer.ppo_transformer import PPOTransformer
    from tiny_transformer.kvtg_integration import KVTGIntegratedTransformer
    from tiny_transformer.seal_integration import SEALIntegratedTransformer

# Define vocabulary
vocab = {str(i): i for i in range(10)}
vocab.update({op: i + 10 for i, op in enumerate(['+', '-', '*', '/'])})
vocab.update({'inf': 14, '<pad>': 15})
rev_vocab = {v: k for k, v in vocab.items()}

# Model hyperparameters
VOCAB_SIZE = len(vocab)
D_MODEL = 64
NHEAD = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 512
MAX_SEQ_LENGTH = 20

def main():
    # Create dataset
    dataset = MathDataset()

    # Create and train standard transformer
    standard_model = StandardTransformer(
        VOCAB_SIZE,
        D_MODEL,
        NHEAD,
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        DIM_FEEDFORWARD,
        MAX_SEQ_LENGTH,
    )
    train_standard_transformer(standard_model, dataset, vocab)
    standard_accuracy = evaluate_standard_transformer(standard_model, dataset, vocab, rev_vocab)
    print(f"Standard Transformer Accuracy: {standard_accuracy}")

    # Create and train PPO transformer
    ppo_model = PPOTransformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH)
    train_ppo_transformer(ppo_model, dataset, vocab, rev_vocab)
    ppo_accuracy = evaluate_ppo_transformer(ppo_model, dataset, vocab, rev_vocab)
    print(f"PPO Transformer Accuracy: {ppo_accuracy}")

    # Create and train KVTG transformer
    kvtg_model = KVTGIntegratedTransformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH, vocab, rev_vocab)
    train_kvtg_transformer(kvtg_model, dataset, vocab)
    kvtg_accuracy = evaluate_kvtg_transformer(kvtg_model, dataset, vocab, rev_vocab)
    print(f"KVTG Transformer Accuracy: {kvtg_accuracy}")

    # Create and train KVTG+SEAL transformer
    seal_model = SEALIntegratedTransformer(
        KVTGIntegratedTransformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH, vocab, rev_vocab),
        vocab,
        rev_vocab,
    )
    train_seal_integrated_transformer(seal_model, dataset, vocab, rev_vocab)
    seal_accuracy = evaluate_seal_integrated_transformer(seal_model, dataset, vocab, rev_vocab)
    print(f"KVTG+SEAL Transformer Accuracy: {seal_accuracy}")

if __name__ == "__main__":
    main()

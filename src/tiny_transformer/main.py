"""Paradigm comparison framework for transformer architectures.

Compares Standard, PPO, KVTG, and KVTG+SEAL transformers on mathematical reasoning.
Each model follows the pattern: test -> train -> test to measure improvement.
KVTG-enabled models include thought graph visualization.
"""

try:  # pragma: no cover - exercised via direct script execution
    from .train import (
        MathDataset,
        train_standard_transformer,
        train_ppo_transformer,
        train_kvtg_transformer,
        train_seal_integrated_transformer,
    )
    from .evaluate_enhanced import (
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
    from tiny_transformer.evaluate_enhanced import (
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
NUM_TEST_PROBLEMS = 5
TRAINING_EPOCHS = 5

def test_train_test_paradigm(model_name: str, model, dataset, vocab, rev_vocab, train_func, eval_func):
    """Execute test->train->test paradigm for a single model."""
    print(f"\n{'='*60}")
    print(f"PARADIGM COMPARISON: {model_name}")
    print(f"{'='*60}")
    
    # Initial test (before training)
    print(f"\n[PHASE 1] Pre-training evaluation on {NUM_TEST_PROBLEMS} problems:")
    pre_accuracy = eval_func(model, dataset, vocab, rev_vocab, num_samples=NUM_TEST_PROBLEMS)
    
    # Training
    print(f"\n[PHASE 2] Training for {TRAINING_EPOCHS} epochs:")
    train_func(model, dataset, vocab, rev_vocab if 'ppo' in model_name.lower() else vocab, epochs=TRAINING_EPOCHS)
    
    # Final test (after training)
    print(f"\n[PHASE 3] Post-training evaluation on {NUM_TEST_PROBLEMS} problems:")
    post_accuracy = eval_func(model, dataset, vocab, rev_vocab, num_samples=NUM_TEST_PROBLEMS)
    
    # Summary
    improvement = post_accuracy - pre_accuracy
    print(f"\n📊 {model_name} SUMMARY:")
    print(f"   Pre-training:  {pre_accuracy:.2%}")
    print(f"   Post-training: {post_accuracy:.2%}")
    print(f"   Improvement:   {improvement:+.2%}")
    
    return pre_accuracy, post_accuracy, improvement

def main():
    """Compare all transformer paradigms using test->train->test framework."""
    print("🧠 TRANSFORMER PARADIGM COMPARISON FRAMEWORK")
    print("Testing: Standard, PPO, KVTG, and KVTG+SEAL architectures")
    print(f"Dataset: {NUM_TEST_PROBLEMS} mathematical problems per test")
    print(f"Training: {TRAINING_EPOCHS} epochs per model\n")
    
    # Create dataset
    dataset = MathDataset(num_samples=1000)
    
    # Results storage
    results = {}
    
    # 1. Standard Transformer
    standard_model = StandardTransformer(
        VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, 
        NUM_DECODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH
    )
    results["Standard"] = test_train_test_paradigm(
        "Standard Transformer", standard_model, dataset, vocab, rev_vocab,
        train_standard_transformer, evaluate_standard_transformer
    )
    
    # 2. PPO Transformer  
    ppo_model = PPOTransformer(
        VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, 
        DIM_FEEDFORWARD, MAX_SEQ_LENGTH
    )
    results["PPO"] = test_train_test_paradigm(
        "PPO Transformer", ppo_model, dataset, vocab, rev_vocab,
        train_ppo_transformer, evaluate_ppo_transformer
    )
    
    # 3. KVTG Transformer
    kvtg_model = KVTGIntegratedTransformer(
        VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, 
        NUM_DECODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH, vocab, rev_vocab
    )
    results["KVTG"] = test_train_test_paradigm(
        "KVTG Transformer", kvtg_model, dataset, vocab, rev_vocab,
        train_kvtg_transformer, evaluate_kvtg_transformer
    )
    
    # 4. KVTG+SEAL Transformer
    seal_base_model = KVTGIntegratedTransformer(
        VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, 
        NUM_DECODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH, vocab, rev_vocab
    )
    seal_model = SEALIntegratedTransformer(seal_base_model, vocab, rev_vocab)
    results["KVTG+SEAL"] = test_train_test_paradigm(
        "KVTG+SEAL Transformer", seal_model, dataset, vocab, rev_vocab,
        train_seal_integrated_transformer, evaluate_seal_integrated_transformer
    )
    
    # Final comparison summary
    print(f"\n{'='*80}")
    print("🏆 FINAL PARADIGM COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Pre-Train':<12} {'Post-Train':<12} {'Improvement':<12}")
    print("-" * 80)
    
    for model_name, (pre, post, improvement) in results.items():
        print(f"{model_name:<15} {pre:<12.2%} {post:<12.2%} {improvement:<+12.2%}")
    
    # Identify best performing model
    best_model = max(results.items(), key=lambda x: x[1][1])  # Best post-training accuracy
    best_improvement = max(results.items(), key=lambda x: x[1][2])  # Best improvement
    
    print(f"\n🥇 Best Overall Performance: {best_model[0]} ({best_model[1][1]:.2%})")
    print(f"📈 Best Improvement: {best_improvement[0]} ({best_improvement[1][2]:+.2%})")
    print(f"\n💡 KVTG models generated thought graph visualizations saved as PNG files.")

if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **KV-Cache Thought Graphs (KVTG)** combined with **Self-Adapting Language Models (SEAL)** - a novel reasoning architecture for LLMs. The system represents reasoning as a directed acyclic graph where each node contains an immutable KV-cache snapshot, enabling non-linear exploration, backtracking, and self-improvement through successful reasoning paths.

## Key Commands

### Dataset Preparation
```bash
# Download datasets (GSM8K and OpenOrca)
python scripts/download_datasets.py

# Process raw datasets into graph format
python src/data_processing/preprocess.py
```

### Training Pipeline
```bash
# Base supervised fine-tuning on graph-structured data
python src/training/train_base.py --model_name mistralai/Mistral-7B-Instruct-v0.1 --dataset_path data/processed/gsm8k_graphs.jsonl --output_dir models/base_sft_model

# SEAL self-improvement training (full implementation)
python src/training/train_seal.py --model_path models/base_sft_model --dataset_path data/processed/gsm8k_graphs.jsonl --output_dir models/seal_model --max_iterations 100 --exploration_budget 20
```

### Testing and Demo
```bash
# Run comprehensive end-to-end tests
python test_e2e.py --verbose --save-results test_results.json

# Interactive demo of KVTG+SEAL system
python demo.py

# Automated demo on all problems
python demo.py --auto
```

### Environment Setup
```bash
pip install -r requirements.txt
# Core dependencies: torch, transformers, accelerate, datasets
```

## Architecture Overview

### Core KVTG System (`src/kvtg/`)
- **`graph.py`**: Defines fundamental data structures (`ThoughtNode`, `ThoughtEdge`, `ThoughtGraph`) that represent reasoning as a graph
- **`controller.py`**: Implements `KVTGController` - the main orchestrator that:
  - Selects promising nodes for expansion using beam search
  - Generates new reasoning steps from KV-cache snapshots
  - Manages graph construction and exploration budget
- **`storage.py`**: Advanced KV-cache storage system with:
  - LRU memory management with disk persistence
  - Hybrid compression (quantization, SVD, sparsification)
  - 8-20x compression ratios with <1e-3 reconstruction error
  - Hardware-aware optimization for GPU memory efficiency

### SEAL Self-Adaptation (`src/seal/`)
- **`adaptation.py`**: Implements `SEALAdapter` that converts successful reasoning paths into fine-tuning data
- Follows MIT's SEAL framework - models learn to generate their own training data from successful solutions

### Data Pipeline
- **`preprocess.py`**: Converts GSM8K math problems and OpenOrca reasoning into graph format with sequential edges
- **`train_base.py`**: Creates training examples where each graph edge becomes a next-step prediction task
- **`train_seal.py`**: Complete SEAL self-improvement pipeline with KVTG exploration and adaptive fine-tuning
- **Dataset format**: JSONL files with `question`, `nodes` (reasoning steps), `edges` (dependencies), and `final_answer`

### Evaluation System (`src/evaluation/`)
- **`math_evaluator.py`**: Robust mathematical reasoning evaluator with:
  - Multi-pattern answer extraction from natural language
  - Numerical comparison with floating-point tolerance
  - Error type classification and confidence scoring
  - Safe expression evaluation for algebraic answers

## Critical Implementation Notes

### KV-Cache Management
The `KVTGStorage` system provides:
- Immutable KV-cache snapshots with unique IDs and LRU eviction
- Advanced compression (quantization, SVD, sparsification) achieving 8-20x space savings
- Automatic disk persistence for large-scale graph exploration
- Perfect reconstruction validation for continued generation

### Graph Construction Logic
- Root node (ID "0") represents initial prompt state
- Child nodes created via beam search with IDs like "{parent_id}-{beam_index}"
- Each node stores: reasoning text, KV-cache ID, and graph position
- Expansion continues until "Final Answer:" detected or budget exhausted

### SEAL Integration Points
- **Success Detection**: Robust `MathEvaluator` with multi-pattern answer extraction and confidence scoring
- **Path Conversion**: `_create_training_example_from_graph()` transforms successful reasoning graphs into SFT data
- **Self-Improvement Loop**: Complete pipeline alternating KVTG exploration and SEAL fine-tuning with statistics tracking

### Model Integration
- Default model: Mistral-7B-Instruct with automatic device mapping
- Tokenizer padding token set to EOS token if not present
- Mixed precision training (fp16) enabled for memory efficiency

## Testing and Quality Assurance

### Comprehensive Test Suite
- **`test_e2e.py`**: End-to-end testing framework covering:
  - KV-cache storage operations and compression
  - ThoughtGraph construction and navigation
  - Mathematical evaluation accuracy
  - Error handling and edge cases
  - Integration workflow validation

### Interactive Demo
- **`demo.py`**: Full-featured demonstration system with:
  - Interactive problem solving interface
  - Real-time KVTG reasoning visualization
  - SEAL adaptation showcasing
  - Custom problem input capabilities

## System Status: Alpha

**The system is currently in an alpha stage.** While the core components for KVTG and SEAL are in place, and a comprehensive safety framework has been developed, the safety protocols are not yet fully integrated into the training and deployment pipelines. The system is ready for experimentation and further development, but not for production use.

**Next Steps:**

- **Run Benchmarks:** Execute the performance benchmarks in `src/benchmarks/` to validate the system's performance claims.
- **Integrate Safety Protocols:** Integrate the `SafetyOrchestrator` from `src/safety/safety_protocols.py` into the training and deployment pipelines.
- **End-to-End Evaluation:** Conduct a thorough end-to-end evaluation of the complete system to assess its performance, stability, and safety.

import torch
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any, Tuple
from .math_env import MathEnv
from .torch_ac.utils.dictlist import DictList
from .utils import tokenize_string, detokenize_tensor

def evaluate_standard_transformer(model, dataset, vocab, rev_vocab, num_samples=5, max_seq_length=20):
    """Evaluate standard transformer on sample problems."""
    model.eval()
    correct = 0
    results = []
    
    # Get first num_samples for consistent testing
    test_samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    
    with torch.no_grad():
        for i, (question, true_answer) in enumerate(test_samples):
            if hasattr(model, 'base_transformer'): # This is a KVTGIntegratedTransformer
                device = model.base_transformer.embedding.weight.device
            else: # This is a StandardTransformer
                device = model.embedding.weight.device
            src = torch.tensor([tokenize_string(question, vocab, max_seq_length)]).to(device)
            tgt = torch.tensor([tokenize_string(true_answer, vocab, max_seq_length)]).to(device)
            
            output = model(src, tgt)
            predicted_answer = detokenize_tensor(output, rev_vocab)
            
            is_correct = _check_math_answer(question, predicted_answer, true_answer)
            if is_correct:
                correct += 1
                
            results.append({
                'question': question,
                'predicted': predicted_answer,
                'true_answer': true_answer,
                'correct': is_correct
            })
    
    accuracy = correct / len(test_samples)
    _print_evaluation_results("Standard Transformer", results, accuracy)
    return accuracy

def evaluate_ppo_transformer(model, dataset, vocab, rev_vocab, num_samples=5):
    """Evaluate PPO transformer on sample problems."""
    model.eval()
    correct = 0
    results = []
    env = MathEnv(vocab, rev_vocab)
    
    test_samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    
    with torch.no_grad():
        for i, (question, true_answer) in enumerate(test_samples):
            # Set environment to specific problem
            env.current_problem = question
            obs, _ = env.reset()
            
            device = next(model.parameters()).device
            obs_tensor = torch.tensor(obs, dtype=torch.long, device=device).unsqueeze(0)
            
            # Generate answer using PPO policy
            generated_tokens = []
            for _ in range(10):  # Max generation length
                dist, _ = model(DictList({"text": obs_tensor}))
                action = dist.sample()
                generated_tokens.append(action.item())
                
                # Update observation (simplified)
                obs_tensor = torch.cat([obs_tensor, action.unsqueeze(0)], dim=1)
                
                if action.item() == vocab.get('<pad>', 15):  # Stop on pad token
                    break
            
            predicted_answer = ''.join([rev_vocab.get(token, '') for token in generated_tokens])
            is_correct = _check_math_answer(question, predicted_answer, true_answer)
            
            if is_correct:
                correct += 1
                
            results.append({
                'question': question,
                'predicted': predicted_answer,
                'true_answer': true_answer,
                'correct': is_correct
            })
    
    accuracy = correct / len(test_samples)
    _print_evaluation_results("PPO Transformer", results, accuracy)
    return accuracy

def evaluate_kvtg_transformer(model, dataset, vocab, rev_vocab, num_samples=5, max_seq_length=20):
    """Evaluate KVTG transformer with thought graph visualization."""
    model.eval()
    correct = 0
    results = []
    
    test_samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    
    with torch.no_grad():
        for i, (question, true_answer) in enumerate(test_samples):
            device = model.base_transformer.embedding.weight.device
            src = torch.tensor([tokenize_string(question, vocab, max_seq_length)]).to(device)
            tgt = torch.tensor([tokenize_string(true_answer, vocab, max_seq_length)]).to(device)
            
            # Use KVTG exploration
            thought_graph = model.kvtg_controller.explore(question=question)
            
            predicted_answer = ""
            if thought_graph and thought_graph.final_answer:
                predicted_answer = thought_graph.final_answer
            else:
                # Fallback to standard generation
                output = model(src, tgt, use_kvtg=False)
                predicted_answer = detokenize_tensor(output, rev_vocab)
            
            is_correct = _check_math_answer(question, predicted_answer, true_answer)
            if is_correct:
                correct += 1
            
            # Visualize thought graph
            if thought_graph:
                _visualize_thought_graph(thought_graph, f"KVTG_Problem_{i+1}")
                
            results.append({
                'question': question,
                'predicted': predicted_answer,
                'true_answer': true_answer,
                'correct': is_correct,
                'thought_graph': thought_graph
            })
    
    accuracy = correct / len(test_samples)
    _print_evaluation_results("KVTG Transformer", results, accuracy)
    return accuracy

def evaluate_seal_integrated_transformer(model, dataset, vocab, rev_vocab, num_samples=5, max_seq_length=20):
    """Evaluate SEAL+KVTG transformer with thought graph visualization."""
    model.eval()
    correct = 0
    results = []
    
    test_samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    
    with torch.no_grad():
        for i, (question, true_answer) in enumerate(test_samples):
            device = model.kvtg_integrated_model.base_transformer.embedding.weight.device
            src = torch.tensor([tokenize_string(question, vocab, max_seq_length)]).to(device)
            tgt = torch.tensor([tokenize_string(true_answer, vocab, max_seq_length)]).to(device)
            
            # Use KVTG exploration through SEAL model
            thought_graph = model.kvtg_integrated_model.kvtg_controller.explore(question=question)
            
            predicted_answer = ""
            if thought_graph and thought_graph.final_answer:
                predicted_answer = thought_graph.final_answer
            else:
                # Fallback to standard generation
                output = model(src, tgt, use_kvtg=False)
                predicted_answer = detokenize_tensor(output, rev_vocab)
            
            is_correct = _check_math_answer(question, predicted_answer, true_answer)
            if is_correct:
                correct += 1
            
            # Visualize thought graph
            if thought_graph:
                _visualize_thought_graph(thought_graph, f"SEAL_Problem_{i+1}")
                
            results.append({
                'question': question,
                'predicted': predicted_answer,
                'true_answer': true_answer,
                'correct': is_correct,
                'thought_graph': thought_graph
            })
    
    accuracy = correct / len(test_samples)
    _print_evaluation_results("SEAL+KVTG Transformer", results, accuracy)
    return accuracy

def _check_math_answer(question: str, predicted: str, true_answer: str) -> bool:
    """Check if predicted answer matches the true mathematical answer."""
    try:
        # Clean predicted answer
        predicted_clean = predicted.strip().replace(' ', '')
        true_clean = true_answer.strip().replace(' ', '')
        
        # Direct string match
        if predicted_clean == true_clean:
            return True
            
        # Try evaluating if they're numeric
        try:
            pred_val = float(predicted_clean) if predicted_clean != 'inf' else float('inf')
            true_val = float(true_clean) if true_clean != 'inf' else float('inf')
            return abs(pred_val - true_val) < 1e-6
        except ValueError:
            pass
            
        # Try evaluating the original question and comparing
        try:
            expected = eval(question)
            predicted_val = eval(predicted_clean) if predicted_clean != 'inf' else float('inf')
            return abs(predicted_val - expected) < 1e-6
        except (SyntaxError, ValueError, ZeroDivisionError):
            pass
            
    except Exception:
        pass
    
    return False

def _visualize_thought_graph(thought_graph, title: str):
    """Visualize KVTG thought graph using networkx and matplotlib."""
    try:
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in thought_graph.nodes.items():
            G.add_node(node_id, text=node.content[:50] + "..." if len(node.content) > 50 else node.content)
        
        # Add edges
        for edge in thought_graph.edges:
            G.add_edge(edge.source_id, edge.target_id)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.6)
        
        # Draw labels
        labels = {node: f"{node}\n{data['text']}" for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"Thought Graph: {title}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"thought_graph_{title.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print graph info
        print(f"\n--- {title} Thought Graph ---")
        print(f"Nodes: {len(thought_graph.nodes)}")
        print(f"Edges: {len(thought_graph.edges)}")
        if thought_graph.final_answer:
            print(f"Final Answer: {thought_graph.final_answer}")
        print("---")
        
    except Exception as e:
        print(f"Error visualizing thought graph for {title}: {e}")

def _print_evaluation_results(model_name: str, results: List[Dict], accuracy: float):
    """Print detailed evaluation results."""
    print(f"\n=== {model_name} Evaluation Results ===")
    print(f"Overall Accuracy: {accuracy:.2%} ({sum(r['correct'] for r in results)}/{len(results)})")
    
    for i, result in enumerate(results, 1):
        status = "✓" if result['correct'] else "✗"
        print(f"{status} Problem {i}: {result['question']} = {result['predicted']} (Expected: {result['true_answer']})")
    
    print("=" * 50)
import logging
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.kvtg.graph import ThoughtGraph, ThoughtNode, ThoughtEdge
from src.kvtg.storage import KVTGStorage, KVCacheType

class KVTGController:
    """
    Orchestrates the exploration of a problem space by building a KV-Cache Thought Graph.

    This controller manages the main loop of:
    1. Selecting promising nodes in the graph to expand.
    2. Generating new thoughts (next steps) from those nodes using the LLM.
    3. Storing the KV-cache of each new thought.
    4. Adding the new thoughts as nodes to the graph.
    5. Repeating until a solution is found or resources are exhausted.
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, storage: KVTGStorage, exploration_budget: int = 50, beam_width: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.storage = storage # Manages KV-cache snapshots
        
        # --- Search Strategy Parameters ---
        self.exploration_budget = exploration_budget # Max number of nodes to generate
        self.beam_width = beam_width # Number of new thoughts to generate from each node

        logging.info("KVTGController initialized.")

    def _format_prompt_for_node(self, graph: ThoughtGraph, node_id: str) -> str:
        """
        Constructs the prompt for generating the next step from a specific node.
        This requires traversing the graph backwards from the given node to the root.
        """
        path_nodes = []
        curr_id = node_id
        # This is a simplified traversal. A real implementation would need
        # to handle complex graph structures (merges, etc.).
        while curr_id != "0": # Assuming "0" is the root/question node
            node = graph.get_node(curr_id)
            if not node:
                break # Should not happen in a well-formed graph
            path_nodes.insert(0, node.text)
            # Simple parent traversal. Will need enhancement for multi-parent nodes.
            parent_edge = next((edge for edge in graph.edges if edge.target == curr_id), None)
            if not parent_edge:
                break
            curr_id = parent_edge.source

        path_str = "\n".join(path_nodes)
        return f"Question: {graph.question}\n\nReasoning Path:\n{path_str}\n\nNext Step:"

    def _generate_next_steps(self, prompt: str, past_key_values: Optional[KVCacheType]) -> Tuple[List[str], List[KVCacheType]]:
        """
        Generates a beam of possible next thoughts from a given state.
        Returns the generated texts and their corresponding KV-cache states.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        # The core generation call.
        # `past_key_values` would be retrieved from self.storage for the parent node.
        # This allows generation to continue from that state without re-computation.
        with torch.no_grad():  # Save memory during inference
            outputs = self.model.generate(
                **inputs,
                past_key_values=past_key_values,
                max_new_tokens=50,
                num_beams=self.beam_width,
                num_return_sequences=self.beam_width,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True, # Needed for return_dict_in_generate
                use_cache=True,  # Enable KV-cache
                do_sample=False,  # Use deterministic beam search
            )

        # Extract the KV-cache from each generated sequence
        # Note: outputs.past_key_values contains the final KV-cache state for each beam
        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            # For beam search, we get one past_key_values per sequence
            if isinstance(outputs.past_key_values, tuple):
                # Single KV-cache returned - replicate for each beam
                new_kv_caches = [outputs.past_key_values] * self.beam_width
            else:
                new_kv_caches = outputs.past_key_values
        else:
            # Fallback: run forward pass to get KV-cache for each sequence
            new_kv_caches = []
            for seq in outputs.sequences:
                seq_inputs = {"input_ids": seq.unsqueeze(0), "attention_mask": torch.ones_like(seq).unsqueeze(0)}
                with torch.no_grad():
                    seq_outputs = self.model(**seq_inputs, use_cache=True)
                new_kv_caches.append(seq_outputs.past_key_values)

        # Decode only the newly generated tokens
        input_length = inputs.input_ids.shape[1]
        generated_sequences = outputs.sequences[:, input_length:]
        next_steps = self.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        
        # Ensure we have the right number of KV-caches
        if len(new_kv_caches) != len(next_steps):
            logging.warning(f"Mismatch: {len(new_kv_caches)} KV-caches for {len(next_steps)} sequences")
            new_kv_caches = new_kv_caches[:len(next_steps)]
        
        return [text.strip() for text in next_steps], new_kv_caches

    def _select_nodes_to_expand(self, graph: ThoughtGraph) -> List[ThoughtNode]:
        """
        Selects the most promising leaf nodes to expand next.
        This is the core of the search strategy (e.g., beam search, best-first).
        Can be overridden by guided exploration systems.
        """
        leaf_nodes = graph.get_leaf_nodes()
        
        # Limit to beam_width to prevent exponential explosion
        if len(leaf_nodes) <= self.beam_width:
            return leaf_nodes
        
        # Simple heuristic: prefer nodes with more mathematical content
        def score_node(node: ThoughtNode) -> float:
            text = node.text.lower()
            score = 0.0
            
            # Bonus for mathematical operations
            import re
            if re.search(r'\d+\s*[+\-*/]\s*\d+', text):
                score += 0.3
            
            # Bonus for equals signs (indicating calculations)
            if '=' in text:
                score += 0.2
            
            # Bonus for reasoning words
            reasoning_words = ['so', 'therefore', 'then', 'next', 'thus']
            for word in reasoning_words:
                if word in text:
                    score += 0.1
                    break
            
            # Penalty for very short or empty responses
            if len(text.strip()) < 5:
                score -= 0.5
            
            # Slight preference for longer, more detailed reasoning
            word_count = len(text.split())
            if 5 <= word_count <= 30:
                score += 0.1
            
            return score
        
        # Score and sort nodes
        scored_nodes = [(node, score_node(node)) for node in leaf_nodes]
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, _ in scored_nodes[:self.beam_width]]

    def explore(self, question: str) -> Optional[ThoughtGraph]:
        """
        Main entry point to start the graph-based reasoning process.
        """
        logging.info(f"Starting KVTG exploration for question: '{question}'")
        graph_id = f"g_{torch.randint(0, 100000, (1,)).item()}"
        graph = ThoughtGraph(id=graph_id, question=question, final_answer="")
        # The root node represents the initial state before any thoughts.
        # It has no text and no KV-cache.
        root_node = ThoughtNode(id="0", text="Start", kv_cache_id=None)
        graph.add_node(root_node)

        nodes_generated = 0
        while nodes_generated < self.exploration_budget:
            nodes_to_expand = self._select_nodes_to_expand(graph)
            if not nodes_to_expand:
                logging.warning("Exploration stalled: no more nodes to expand.")
                break

            for node in nodes_to_expand:
                if nodes_generated >= self.exploration_budget: break

                # Fetch the KV-cache for the parent node. For the root node, this will be None.
                parent_kv_cache = self.storage.get(node.kv_cache_id) if node.kv_cache_id else None
                prompt = self._format_prompt_for_node(graph, node.id)
                
                new_thoughts, new_kv_caches = self._generate_next_steps(prompt, parent_kv_cache)

                for i, (thought_text, kv_cache) in enumerate(zip(new_thoughts, new_kv_caches)):
                    if nodes_generated >= self.exploration_budget: break

                    new_node_id = f"{node.id}-{i}"
                    kv_cache_id = f"{graph.id}-{new_node_id}"
                    self.storage.store(kv_cache_id, kv_cache)

                    new_node = ThoughtNode(id=new_node_id, text=thought_text, kv_cache_id=kv_cache_id)
                    graph.add_node(new_node)
                    graph.add_edge(ThoughtEdge(source=node.id, target=new_node.id))
                    nodes_generated += 1

                    # Check for a solution
                    if "Final Answer:" in thought_text:
                        logging.info(f"Solution found at node {new_node_id}. Stopping exploration.")
                        graph.final_answer = thought_text # Or parse it more carefully
                        return graph
            
        logging.info(f"Exploration budget of {self.exploration_budget} nodes reached.")
        return graph
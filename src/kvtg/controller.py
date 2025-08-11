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
            output_past=True, # We need the new KV cache
        )

        # The new KV-cache for each generated sequence needs to be captured.
        new_kv_caches = outputs.past_key_values

        # We need to decode only the newly generated tokens, not the input prompt.
        input_length = inputs.input_ids.shape[1]
        generated_sequences = outputs.sequences[:, input_length:]
        next_steps = self.tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        return [text.strip() for text in next_steps], new_kv_caches

    def _select_nodes_to_expand(self, graph: ThoughtGraph) -> List[ThoughtNode]:
        """
        Selects the most promising leaf nodes to expand next.
        This is the core of the search strategy (e.g., beam search, best-first).
        A simple strategy is to expand all current leaf nodes.
        A more advanced one would score nodes and pick the top N.
        """
        # For now, a simple strategy: expand all leaf nodes.
        return graph.get_leaf_nodes()

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
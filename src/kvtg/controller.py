import logging
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.kvtg.graph import ThoughtGraph, ThoughtNode, ThoughtEdge
# from src.kvtg.storage import KVTGStorage # To be implemented

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
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, exploration_budget: int = 50, beam_width: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        # self.storage = KVTGStorage(device=self.device) # Manages KV-cache snapshots
        
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

    def _generate_next_steps(self, prompt: str, past_key_values: Optional[torch.Tensor]) -> List[str]:
        """
        Generates a beam of possible next thoughts from a given state.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
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
            output_past=True # We need the new KV cache
        )

        # TODO: The new KV-cache for each generated sequence needs to be captured
        # and stored in self.storage. This is a key part of the KVTG process.

        generated_texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        # We only want the newly generated part, not the whole prompt.
        next_steps = [text[len(prompt):].strip() for text in generated_texts]
        return next_steps

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
        graph = ThoughtGraph(id="g1", question=question, final_answer="") # Simplified ID
        # The root node represents the initial state before any thoughts.
        root_node = ThoughtNode(id="0", text="Start")
        graph.add_node(root_node)

        for i in range(self.exploration_budget):
            nodes_to_expand = self._select_nodes_to_expand(graph)
            if not nodes_to_expand:
                logging.warning("Exploration stalled: no more nodes to expand.")
                break

            for node in nodes_to_expand:
                # In a real implementation, we'd fetch the KV-cache for `node.id` here.
                # parent_kv_cache = self.storage.get(node.id)
                prompt = self._format_prompt_for_node(graph, node.id)
                
                # new_thoughts = self._generate_next_steps(prompt, parent_kv_cache)
                # For now, we pass None for the cache.
                new_thoughts = self._generate_next_steps(prompt, None)

                # TODO: Add new thoughts as nodes to the graph, linking them to the parent.
                # TODO: Check if any new thought represents a complete solution.

        logging.info("Exploration budget reached.")
        return graph
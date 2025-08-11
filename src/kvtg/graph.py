
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ThoughtNode:
    """Represents a single node in the thought graph."""
    id: str
    text: str
    # In the full KVTG implementation, this would also hold kv_ptr, embeddings, etc.

@dataclass
class ThoughtEdge:
    """Represents a directed edge between two nodes in the thought graph."""
    source: str  # ID of the source node
    target: str  # ID of the target node
    type: str = 'sequential' # e.g., sequential, supports, contradicts

@dataclass
class ThoughtGraph:
    """Represents a complete thought graph for a single problem."""
    id: str
    question: str
    final_answer: str
    nodes: List[ThoughtNode] = field(default_factory=list)
    edges: List[ThoughtEdge] = field(default_factory=list)

    def add_node(self, node: ThoughtNode):
        """Adds a node to the graph."""
        if not self.get_node(node.id):
            self.nodes.append(node)

    def add_edge(self, edge: ThoughtEdge):
        """Adds an edge to the graph."""
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        """Retrieves a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_leaf_nodes(self) -> List[ThoughtNode]:
        """Returns all nodes that are not sources of any edge (leaf nodes)."""
        source_ids = {edge.source for edge in self.edges}
        # We also exclude the root node '0' from being a leaf if it's the only node.
        leaf_nodes = [node for node in self.nodes if node.id not in source_ids and (node.id != '0' or len(self.nodes) > 1)]
        return leaf_nodes

    def get_terminal_nodes(self) -> List[ThoughtNode]:
        """Placeholder for finding nodes that represent a final answer."""
        # This logic will need to be more sophisticated, e.g., checking for a special token.
        return [node for node in self.nodes if "Final Answer:" in node.text]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThoughtGraph':
        """Creates a ThoughtGraph instance from a dictionary (from JSONL)."""
        nodes = [ThoughtNode(**node_data) for node_data in data.get('nodes', [])]
        edges = [ThoughtEdge(**edge_data) for edge_data in data.get('edges', [])]
        
        return cls(
            id=data['id'],
            question=data['question'],
            final_answer=data['final_answer'],
            nodes=nodes,
            edges=edges
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ThoughtGraph instance back to a dictionary."""
        return {
            'id': self.id,
            'question': self.question,
            'final_answer': self.final_answer,
            'nodes': [node.__dict__ for node in self.nodes],
            'edges': [edge.__dict__ for edge in self.edges]
        }

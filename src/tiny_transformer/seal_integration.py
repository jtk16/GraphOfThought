import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Dict, Any

# Add the parent directory of src to the Python path to import seal
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.seal.adaptation import SEALAdapter
from src.kvtg.graph import ThoughtGraph

class SEALIntegratedTransformer(nn.Module):
    def __init__(self, kvtg_integrated_model, vocab, rev_vocab):
        super(SEALIntegratedTransformer, self).__init__()
        self.kvtg_integrated_model = kvtg_integrated_model
        self.vocab = vocab
        self.rev_vocab = rev_vocab

        # Initialize SEALAdapter
        # SEALAdapter expects a model and tokenizer that can be fine-tuned
        # We'll use the mock model and tokenizer from KVTGIntegratedTransformer
        self.seal_adapter = SEALAdapter(
            model=self.kvtg_integrated_model.mock_model,
            tokenizer=self.kvtg_integrated_model.mock_tokenizer,
            output_dir="./seal_output" # Dummy output directory for this test
        )

    def forward(self, src, tgt, use_kvtg=False, problem_id=None, is_successful=False):
        # First, run the KVTG-integrated model
        output = self.kvtg_integrated_model(src, tgt, use_kvtg=use_kvtg, problem_id=problem_id)

        # If this is a training step and the problem was successful, adapt the model
        if is_successful and use_kvtg and problem_id is not None:
            # In a real scenario, we'd need the actual ThoughtGraph that led to success
            # For this simplified test, we'll just create a dummy ThoughtGraph
            # and assume the output represents a successful path.
            
            # This part needs to be carefully designed to get the actual thought graph
            # from the KVTG exploration that led to the successful answer.
            # For now, we'll just create a dummy graph with the question and answer.
            
            # Convert src to question string
            question_str = self.kvtg_integrated_model.mock_tokenizer.batch_decode(src, skip_special_tokens=False)[0]
            # Convert output logits to predicted answer string
            predicted_answer_str = self.kvtg_integrated_model.mock_tokenizer.batch_decode(torch.argmax(output, dim=-1), skip_special_tokens=True)[0]

            dummy_graph = ThoughtGraph(
                id=problem_id,
                question=question_str,
                final_answer=predicted_answer_str # Assuming this is the successful answer
            )
            
            # Fine-tune the model using SEAL
            self.seal_adapter.finetune_on_path(dummy_graph)

        return output

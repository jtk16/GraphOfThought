#!/usr/bin/env python3
"""
Data validation and fixing script for KVTG datasets.
Ensures proper formatting and consistency with the KVTG architecture.
"""

import json
import logging
import os
import sys
from typing import Dict, Any, List, Tuple
import re

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.kvtg.graph import ThoughtGraph, ThoughtNode, ThoughtEdge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validates and fixes KVTG dataset formatting."""
    
    def __init__(self):
        self.validation_stats = {
            'total_records': 0,
            'valid_records': 0,
            'fixed_records': 0,
            'discarded_records': 0,
            'issues_found': []
        }
    
    def validate_single_record(self, record: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Validate and fix a single record.
        
        Returns:
            (is_valid, fixed_record, issues_found)
        """
        issues = []
        fixed_record = record.copy()
        
        # Check required fields
        required_fields = ['id', 'question', 'final_answer', 'nodes', 'edges']
        for field in required_fields:
            if field not in record:
                issues.append(f"Missing required field: {field}")
                if field == 'nodes':
                    fixed_record[field] = []
                elif field == 'edges':
                    fixed_record[field] = []
                else:
                    fixed_record[field] = ""
        
        # Validate and fix question
        if not fixed_record.get('question') or not fixed_record['question'].strip():
            issues.append("Empty or missing question")
            return False, fixed_record, issues  # Cannot fix empty questions
        
        # Clean and validate question
        question = fixed_record['question'].strip()
        if len(question) < 10:
            issues.append("Question too short (likely invalid)")
            return False, fixed_record, issues
        
        # Validate nodes structure
        nodes = fixed_record.get('nodes', [])
        if not isinstance(nodes, list):
            issues.append("Nodes field is not a list")
            nodes = []
            fixed_record['nodes'] = nodes
        
        # Fix and validate individual nodes
        valid_nodes = []
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                issues.append(f"Node {i} is not a dictionary")
                continue
            
            # Ensure required node fields
            if 'id' not in node:
                node['id'] = f"step_{i+1}"
                issues.append(f"Added missing ID to node {i}")
            
            if 'text' not in node or not node['text'].strip():
                issues.append(f"Node {i} has empty or missing text")
                continue  # Skip empty nodes
            
            # Clean node text
            node['text'] = node['text'].strip()
            
            # Remove any HTML-like tags that might have leaked through
            node['text'] = re.sub(r'<<[^>]*>>', '', node['text'])
            node['text'] = re.sub(r'<[^>]*>', '', node['text'])
            
            valid_nodes.append(node)
        
        fixed_record['nodes'] = valid_nodes
        
        # If no valid nodes, this is not a useful record
        if not valid_nodes:
            issues.append("No valid reasoning nodes found")
            return False, fixed_record, issues
        
        # Validate edges structure
        edges = fixed_record.get('edges', [])
        if not isinstance(edges, list):
            issues.append("Edges field is not a list")
            edges = []
        
        # Create proper sequential edges if missing/invalid
        valid_edges = []
        valid_node_ids = [node['id'] for node in valid_nodes]
        
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            
            source = edge.get('source')
            target = edge.get('target')
            
            if source in valid_node_ids and target in valid_node_ids:
                if 'type' not in edge:
                    edge['type'] = 'sequential'
                valid_edges.append(edge)
        
        # If no valid edges, create sequential chain
        if not valid_edges and len(valid_nodes) > 1:
            for i in range(len(valid_nodes) - 1):
                valid_edges.append({
                    'source': valid_nodes[i]['id'],
                    'target': valid_nodes[i+1]['id'],
                    'type': 'sequential'
                })
            issues.append("Created missing sequential edges")
        
        fixed_record['edges'] = valid_edges
        
        # Validate final answer
        final_answer = fixed_record.get('final_answer', '').strip()
        
        # Try to extract final answer from the last node if missing
        if not final_answer and valid_nodes:
            last_node_text = valid_nodes[-1]['text']
            
            # Look for explicit final answer patterns
            final_patterns = [
                r'final answer[:\s]*(.+?)(?:\.|$)',
                r'answer[:\s]*(.+?)(?:\.|$)',
                r'therefore[:\s]*(.+?)(?:\.|$)',
                r'so[:\s]*(.+?)(?:\.|$)',
                r'=\s*([^.]+?)(?:\.|$)'
            ]
            
            for pattern in final_patterns:
                match = re.search(pattern, last_node_text, re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    # Clean up extracted answer
                    extracted = re.sub(r'[^\w\d./\-$%]', '', extracted)
                    if extracted and len(extracted) <= 20:  # Reasonable answer length
                        fixed_record['final_answer'] = extracted
                        issues.append(f"Extracted final answer from reasoning: {extracted}")
                        break
        
        # Final validation
        is_valid = (
            bool(fixed_record['question'].strip()) and
            bool(fixed_record['nodes']) and
            len(fixed_record['nodes']) >= 1
        )
        
        return is_valid, fixed_record, issues
    
    def process_dataset(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Process an entire dataset file."""
        logger.info(f"Processing dataset: {input_path}")
        
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return {'error': 'File not found'}
        
        valid_records = []
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        record = json.loads(line)
                        self.validation_stats['total_records'] += 1
                        
                        is_valid, fixed_record, issues = self.validate_single_record(record)
                        
                        if issues:
                            self.validation_stats['issues_found'].extend([
                                f"Line {line_num}: {issue}" for issue in issues
                            ])
                        
                        if is_valid:
                            self.validation_stats['valid_records'] += 1
                            if issues:  # Was fixed
                                self.validation_stats['fixed_records'] += 1
                            valid_records.append(fixed_record)
                        else:
                            self.validation_stats['discarded_records'] += 1
                            logger.warning(f"Discarded record at line {line_num}: {issues}")
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error at line {line_num}: {e}")
                        self.validation_stats['discarded_records'] += 1
        
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            return {'error': str(e)}
        
        # Write cleaned data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in valid_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Processed dataset saved to: {output_path}")
        return {
            'input_file': input_path,
            'output_file': output_path,
            'statistics': self.validation_stats,
            'valid_records_count': len(valid_records)
        }
    
    def generate_sample_records(self, count: int = 5) -> List[Dict[str, Any]]:
        """Generate well-formatted sample records for testing."""
        samples = [
            {
                'id': 'sample_math_1',
                'question': 'Sarah has 15 apples. She gives 7 to her friend and buys 12 more. How many apples does Sarah have now?',
                'final_answer': '20',
                'nodes': [
                    {'id': 'step_1', 'text': 'Sarah starts with 15 apples.'},
                    {'id': 'step_2', 'text': 'She gives away 7 apples, so she has 15 - 7 = 8 apples left.'},
                    {'id': 'step_3', 'text': 'She buys 12 more apples, so she has 8 + 12 = 20 apples.'},
                    {'id': 'step_4', 'text': 'Final Answer: Sarah has 20 apples.'}
                ],
                'edges': [
                    {'source': 'step_1', 'target': 'step_2', 'type': 'sequential'},
                    {'source': 'step_2', 'target': 'step_3', 'type': 'sequential'},
                    {'source': 'step_3', 'target': 'step_4', 'type': 'sequential'}
                ]
            },
            {
                'id': 'sample_math_2',
                'question': 'What is 24 × 15?',
                'final_answer': '360',
                'nodes': [
                    {'id': 'step_1', 'text': 'I need to calculate 24 × 15.'},
                    {'id': 'step_2', 'text': 'I can break this down: 24 × 15 = 24 × (10 + 5) = (24 × 10) + (24 × 5).'},
                    {'id': 'step_3', 'text': '24 × 10 = 240'},
                    {'id': 'step_4', 'text': '24 × 5 = 120'},
                    {'id': 'step_5', 'text': 'Therefore: 240 + 120 = 360'}
                ],
                'edges': [
                    {'source': 'step_1', 'target': 'step_2', 'type': 'sequential'},
                    {'source': 'step_2', 'target': 'step_3', 'type': 'sequential'},
                    {'source': 'step_2', 'target': 'step_4', 'type': 'sequential'},
                    {'source': 'step_3', 'target': 'step_5', 'type': 'sequential'},
                    {'source': 'step_4', 'target': 'step_5', 'type': 'sequential'}
                ]
            },
            {
                'id': 'sample_word_problem_1',
                'question': 'A pizza is cut into 8 equal slices. Tom eats 3 slices and Jerry eats 2 slices. What fraction of the pizza is left?',
                'final_answer': '3/8',
                'nodes': [
                    {'id': 'step_1', 'text': 'The pizza has 8 equal slices total.'},
                    {'id': 'step_2', 'text': 'Tom eats 3 slices and Jerry eats 2 slices.'},
                    {'id': 'step_3', 'text': 'Total slices eaten: 3 + 2 = 5 slices.'},
                    {'id': 'step_4', 'text': 'Slices remaining: 8 - 5 = 3 slices.'},
                    {'id': 'step_5', 'text': 'Fraction left: 3 out of 8 slices = 3/8.'}
                ],
                'edges': [
                    {'source': 'step_1', 'target': 'step_2', 'type': 'sequential'},
                    {'source': 'step_2', 'target': 'step_3', 'type': 'sequential'},
                    {'source': 'step_3', 'target': 'step_4', 'type': 'sequential'},
                    {'source': 'step_4', 'target': 'step_5', 'type': 'sequential'}
                ]
            }
        ]
        
        return samples[:count]

def main():
    """Main function to process datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and fix KVTG datasets")
    parser.add_argument("--input_dir", type=str, default="data/processed", help="Input directory")
    parser.add_argument("--output_dir", type=str, default="data/validated", help="Output directory")
    parser.add_argument("--generate_samples", action="store_true", help="Generate sample data")
    parser.add_argument("--validate_only", action="store_true", help="Only validate, don't write output")
    
    args = parser.parse_args()
    
    validator = DataValidator()
    
    # Generate samples if requested
    if args.generate_samples:
        samples = validator.generate_sample_records()
        sample_path = os.path.join(args.output_dir, "sample_problems.jsonl")
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(sample_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Generated {len(samples)} sample records in {sample_path}")
    
    # Process existing datasets
    datasets = [
        ("gsm8k_graphs.jsonl", "gsm8k_validated.jsonl"),
        ("openorca_graphs.jsonl", "openorca_validated.jsonl")
    ]
    
    total_results = []
    
    for input_file, output_file in datasets:
        input_path = os.path.join(args.input_dir, input_file)
        output_path = os.path.join(args.output_dir, output_file)
        
        if os.path.exists(input_path):
            if args.validate_only:
                logger.info(f"Validating {input_path} (no output)...")
                # Just validate without writing
                result = validator.process_dataset(input_path, "/dev/null")
            else:
                result = validator.process_dataset(input_path, output_path)
            total_results.append(result)
        else:
            logger.warning(f"Input file not found: {input_path}")
    
    # Summary report
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY REPORT")
    logger.info("="*60)
    
    for result in total_results:
        if 'error' in result:
            logger.error(f"Error processing {result.get('input_file', 'unknown')}: {result['error']}")
            continue
        
        stats = result['statistics']
        logger.info(f"\nFile: {result['input_file']}")
        logger.info(f"  Total records: {stats['total_records']}")
        logger.info(f"  Valid records: {stats['valid_records']}")
        logger.info(f"  Fixed records: {stats['fixed_records']}")
        logger.info(f"  Discarded records: {stats['discarded_records']}")
        logger.info(f"  Success rate: {stats['valid_records']/max(1, stats['total_records'])*100:.1f}%")
        
        if stats['issues_found']:
            logger.info(f"  Issues found: {len(stats['issues_found'])}")
            # Show first few issues as examples
            for issue in stats['issues_found'][:5]:
                logger.info(f"    - {issue}")
            if len(stats['issues_found']) > 5:
                logger.info(f"    ... and {len(stats['issues_found'])-5} more issues")

if __name__ == "__main__":
    main()
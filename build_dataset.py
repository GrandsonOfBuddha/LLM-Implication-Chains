import os
import json
import jsonlines
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import time
from tqdm import tqdm
from implication_chains import (
    ImplicationChainGenerator, 
    ChainGraphBuilder, 
    PairSampler, 
    create_seed_statements
)


class DatasetBuilder:
    """Builds the complete dataset from seed statements"""
    
    def __init__(self, openai_api_key: str, chain_length: int = 3, min_distance: int = 2):
        self.generator = ImplicationChainGenerator(openai_api_key)
        self.graph_builder = ChainGraphBuilder()
        self.sampler = PairSampler(min_distance=min_distance)
        self.chain_length = chain_length
    
    def build_dataset(self, seed_statements: List[str], 
                     output_file: str = "dataset.jsonl",
                     target_pairs: int = 5000) -> Dict:
        """Build the complete dataset from seeds to labeled pairs"""
        
        print(f"Generating chains from {len(seed_statements)} seeds...")
        chains = []
        
        for i, seed in enumerate(tqdm(seed_statements)):
            try:
                # Each seed gets a fresh, independent chain generation
                print(f"\nGenerating chain {i+1}/{len(seed_statements)} from seed: '{seed[:50]}{'...' if len(seed) > 50 else ''}'")
                
                chain = self.generator.generate_chain(seed, self.chain_length)
                
                # Validate chain coherence
                validation = self.generator.validate_chain_coherence(chain)
                if not validation["is_coherent"]:
                    print(f"Warning: Chain coherence issues detected: {validation['issues']}")
                
                chains.append(chain)
                
                # Log the generated chain for verification
                print(f"Generated chain with {len(chain)} statements:")
                for j, stmt in enumerate(chain):
                    print(f"  {j}: {stmt.text}")
                
                # Save intermediate results every 10 chains
                if (i + 1) % 10 == 0:
                    self.save_chains(chains, f"chains_checkpoint_{i+1}.json")
                    print(f"Saved checkpoint after {i+1} chains")
                    
            except Exception as e:
                print(f"Error processing seed '{seed}': {e}")
                continue
        
        print(f"\nGenerated {len(chains)} chains successfully")
        
        # Sample pairs from chains
        print("Sampling pairs...")
        all_pairs = []
        
        for chain in tqdm(chains):
            chain_pairs = self.sampler.sample_pairs_from_chain(chain)
            all_pairs.extend(chain_pairs)
        
        # Add cross-chain pairs for independence/contradiction
        cross_pairs = self.sampler.sample_cross_chain_pairs(
            chains, num_pairs=min(1000, target_pairs // 5)
        )
        all_pairs.extend(cross_pairs)
        
        # Shuffle and limit to target size
        import random
        random.shuffle(all_pairs)
        all_pairs = all_pairs[:target_pairs]
        
        # Save dataset
        self.save_dataset(all_pairs, output_file)
        
        # Generate statistics
        stats = self.generate_statistics(all_pairs, chains)
        
        return {
            "chains": chains,
            "pairs": all_pairs,
            "stats": stats
        }
    
    def save_chains(self, chains: List, filename: str):
        """Save chains to JSON file with validation results"""
        chains_data = []
        for i, chain in enumerate(chains):
            # Get validation results for this chain
            validation = self.generator.validate_chain_coherence(chain) if hasattr(self.generator, 'validate_chain_coherence') else {"is_coherent": True, "issues": []}
            
            chain_data = {
                "chain_id": i,
                "seed_statement": chain[0].text if chain else "",
                "validation": validation,
                "statements": [
                    {
                        "text": stmt.text,
                        "atoms": [asdict(atom) for atom in stmt.atoms],
                        "id": stmt.id
                    } for stmt in chain
                ]
            }
            chains_data.append(chain_data)
        
        with open(filename, 'w') as f:
            json.dump(chains_data, f, indent=2)
        
        # Print validation summary
        coherent_chains = sum(1 for chain_data in chains_data if chain_data["validation"]["is_coherent"])
        print(f"Saved {len(chains)} chains to {filename} ({coherent_chains}/{len(chains)} coherent)")
    
    def save_dataset(self, pairs: List[Dict], filename: str):
        """Save dataset to JSONL file"""
        with jsonlines.open(filename, 'w') as writer:
            for pair in pairs:
                writer.write(pair)
        
        print(f"Saved {len(pairs)} pairs to {filename}")
    
    def generate_statistics(self, pairs: List[Dict], chains: List) -> Dict:
        """Generate dataset statistics"""
        label_counts = {}
        for pair in pairs:
            label = pair["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        stats = {
            "total_pairs": len(pairs),
            "label_distribution": label_counts,
            "total_chains": len(chains),
            "avg_chain_length": sum(len(chain) for chain in chains) / len(chains) if chains else 0,
            "total_statements": sum(len(chain) for chain in chains)
        }
        
        # Save detailed statistics to text file
        with open("dataset_statistics.txt", 'w') as f:
            f.write("MVP IMPLICATION CHAINS - DATASET STATISTICS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {__import__('datetime').datetime.now()}\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total pairs: {stats['total_pairs']}\n")
            f.write(f"Total chains: {stats['total_chains']}\n")
            f.write(f"Total statements: {stats['total_statements']}\n")
            f.write(f"Average chain length: {stats['avg_chain_length']:.2f}\n\n")
            
            f.write("LABEL DISTRIBUTION\n")
            f.write("-" * 20 + "\n")
            for label, count in label_counts.items():
                percentage = (count / len(pairs)) * 100
                f.write(f"{label:<12}: {count:>5} ({percentage:>5.1f}%)\n")
            
            f.write("\nCHAIN DETAILS\n")
            f.write("-" * 20 + "\n")
            for i, chain in enumerate(chains[:10]):  # Show first 10 chains as examples
                f.write(f"Chain {i+1} ({len(chain)} statements):\n")
                for j, stmt in enumerate(chain):
                    f.write(f"  {j}: {stmt.text}\n")
                f.write("\n")
            
            if len(chains) > 10:
                f.write(f"... and {len(chains) - 10} more chains\n")
        
        print("Dataset Statistics:")
        print(f"Total pairs: {stats['total_pairs']}")
        print(f"Label distribution: {stats['label_distribution']}")
        print(f"Total chains: {stats['total_chains']}")
        print(f"Average chain length: {stats['avg_chain_length']:.2f}")
        print("Detailed statistics saved to: dataset_statistics.txt")
        
        return stats


def load_api_key_from_file() -> str:
    """Load OpenAI API key from KEY.txt file"""
    try:
        with open("KEY.txt", 'r') as f:
            api_key = f.read().strip()
        if not api_key:
            raise ValueError("KEY.txt file is empty")
        return api_key
    except FileNotFoundError:
        raise FileNotFoundError("KEY.txt file not found. Please create it with your OpenAI API key.")


def main():
    """Main function to build the MVP dataset"""
    
    # Configuration
    try:
        OPENAI_API_KEY = load_api_key_from_file()
    except Exception as e:
        print(f"Error loading API key: {e}")
        return
    
    CHAIN_LENGTH = 3
    MIN_DISTANCE = 2
    TARGET_PAIRS = 5000
    NUM_SEEDS = 50  # Use first 50 seeds for MVP
    
    # Get seed statements
    seeds = create_seed_statements()[:NUM_SEEDS]
    
    # Build dataset
    builder = DatasetBuilder(
        openai_api_key=OPENAI_API_KEY,
        chain_length=CHAIN_LENGTH,
        min_distance=MIN_DISTANCE
    )
    
    result = builder.build_dataset(
        seed_statements=seeds,
        output_file="mvp_dataset.jsonl",
        target_pairs=TARGET_PAIRS
    )
    
    # Save additional outputs
    builder.save_chains(result["chains"], "implication_chains.json")
    
    with open("dataset_stats.json", 'w') as f:
        json.dump(result["stats"], f, indent=2)
    
    print("MVP dataset generation complete!")


if __name__ == "__main__":
    main()
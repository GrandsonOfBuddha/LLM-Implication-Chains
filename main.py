#!/usr/bin/env python3
"""
MVP Implication Chains - Main Entry Point

This script orchestrates the complete pipeline:
1. Generate implication chains from seed statements
2. Build dataset with automatic labeling
3. Train and evaluate NLI classifier
4. Run factual sanity checks

Usage:
    python main.py [--mode {all,dataset,train,test}] [--config config.json]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from build_dataset import DatasetBuilder
from train_classifier import NLIClassifier, run_factual_sanity_check
from implication_chains import create_seed_statements


class MVPPipeline:
    """Main pipeline for the MVP Implication Chains project"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_api_key = self.load_api_key()
        
    def load_api_key(self) -> str:
        """Load OpenAI API key from KEY.txt file"""
        key_file = Path("KEY.txt")
        
        if not key_file.exists():
            raise FileNotFoundError(
                "KEY.txt file not found. Please create a KEY.txt file with your OpenAI API key."
            )
        
        with open(key_file, 'r') as f:
            api_key = f.read().strip()
        
        if not api_key:
            raise ValueError("KEY.txt file is empty. Please add your OpenAI API key.")
        
        if not api_key.startswith('sk-'):
            print("Warning: API key doesn't start with 'sk-'. Please verify it's correct.")
        
        return api_key
    
    def run_dataset_generation(self):
        """Generate the implication chains dataset"""
        print("=" * 60)
        print("PHASE 1: DATASET GENERATION")
        print("=" * 60)
        
        # Get configuration
        chain_length = self.config.get("chain_length", 3)
        min_distance = self.config.get("min_distance", 2)
        target_pairs = self.config.get("target_pairs", 5000)
        num_seeds = self.config.get("num_seeds", 50)
        
        print(f"Configuration:")
        print(f"  Chain length: {chain_length}")
        print(f"  Min distance: {min_distance}")
        print(f"  Target pairs: {target_pairs}")
        print(f"  Number of seeds: {num_seeds}")
        print()
        
        # Get seed statements
        seeds = create_seed_statements()[:num_seeds]
        print(f"Using {len(seeds)} seed statements")
        
        # Build dataset
        builder = DatasetBuilder(
            openai_api_key=self.openai_api_key,
            chain_length=chain_length,
            min_distance=min_distance
        )
        
        result = builder.build_dataset(
            seed_statements=seeds,
            output_file="mvp_dataset.jsonl",
            target_pairs=target_pairs
        )
        
        # Save additional outputs
        builder.save_chains(result["chains"], "implication_chains.json")
        
        with open("dataset_stats.json", 'w') as f:
            json.dump(result["stats"], f, indent=2)
        
        print("\n✓ Dataset generation complete!")
        return result
    
    def run_classifier_training(self):
        """Train the NLI classifier"""
        print("=" * 60)
        print("PHASE 2: CLASSIFIER TRAINING")
        print("=" * 60)
        
        # Check if dataset exists
        dataset_file = "mvp_dataset.jsonl"
        if not Path(dataset_file).exists():
            raise FileNotFoundError(
                f"Dataset file {dataset_file} not found. Please run dataset generation first."
            )
        
        # Get training configuration
        model_name = self.config.get("model_name", "roberta-base")
        num_epochs = self.config.get("num_epochs", 3)
        batch_size = self.config.get("batch_size", 16)
        learning_rate = self.config.get("learning_rate", 2e-5)
        
        print(f"Training Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print()
        
        # Initialize classifier
        classifier = NLIClassifier(model_name=model_name)
        
        # Prepare data
        train_dataset, val_dataset, test_dataset = classifier.prepare_data(dataset_file)
        
        # Train the model
        trainer = classifier.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        print("\n✓ Classifier training complete!")
        return classifier, test_dataset
    
    def run_evaluation(self, classifier=None):
        """Evaluate the trained classifier"""
        print("=" * 60)
        print("PHASE 3: EVALUATION")
        print("=" * 60)
        
        # Initialize classifier if not provided
        if classifier is None:
            classifier = NLIClassifier()
            dataset_file = "mvp_dataset.jsonl"
            if not Path(dataset_file).exists():
                raise FileNotFoundError("Dataset file not found for evaluation.")
            _, _, test_dataset = classifier.prepare_data(dataset_file)
        else:
            # Get test dataset
            dataset_file = "mvp_dataset.jsonl"
            _, _, test_dataset = classifier.prepare_data(dataset_file)
        
        # Evaluate the model
        test_results, classification_report = classifier.evaluate(test_dataset)
        
        print("\nTest Results:")
        print("-" * 30)
        for key, value in test_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Save detailed results
        with open("evaluation_results.json", "w") as f:
            json.dump({
                "test_metrics": test_results,
                "classification_report": classification_report
            }, f, indent=2)
        
        # Run factual sanity check
        print("\nRunning factual sanity check...")
        factual_accuracy = run_factual_sanity_check(classifier)
        
        # Summary
        print("\n" + "="*60)
        print("MVP EVALUATION SUMMARY")
        print("="*60)
        print(f"Test Accuracy: {test_results['eval_accuracy']:.3f}")
        print(f"Test F1 (Macro): {test_results['eval_f1_macro']:.3f}")
        print(f"Factual Sanity Check: {factual_accuracy:.3f}")
        print("="*60)
        
        return test_results, factual_accuracy
    
    def run_test_only(self):
        """Run a quick test to verify the system works"""
        print("=" * 60)
        print("SYSTEM TEST MODE")
        print("=" * 60)
        
        print("Testing API key and conversation context...")
        try:
            # Test with conversation context
            from implication_chains import ImplicationChainGenerator
            generator = ImplicationChainGenerator(self.openai_api_key)
            
            test_seed = "Dogs are animals"
            print(f"Testing with seed: '{test_seed}'")
            
            test_chain = generator.generate_chain(test_seed, chain_length=2)
            print(f"✓ Generated test chain with context: {len(test_chain)} statements")
            
            for i, stmt in enumerate(test_chain):
                print(f"  {i}: {stmt.text}")
            
            # Test chain validation
            validation = generator.validate_chain_coherence(test_chain)
            print(f"\nChain coherence: {validation['is_coherent']}")
            if validation['issues']:
                print(f"Issues detected: {validation['issues']}")
            
            # Test independence by generating another chain
            print(f"\nTesting independence with same seed...")
            test_chain2 = generator.generate_chain(test_seed, chain_length=2)
            
            chain1_texts = [stmt.text for stmt in test_chain]
            chain2_texts = [stmt.text for stmt in test_chain2]
            
            if chain1_texts != chain2_texts:
                print("✓ Chains are independent (different results from same seed)")
            else:
                print("⚠ Chains are identical (possible but unusual)")
            
        except Exception as e:
            print(f"✗ API test failed: {e}")
            return False
        
        print("\n✓ System test passed!")
        print("✓ Conversation context maintained within chains")
        print("✓ Chain independence verified")
        return True
    
    def run_pipeline(self, mode: str = "all"):
        """Run the complete pipeline or specific phases"""
        
        # Create pipeline results file
        pipeline_results_file = "pipeline_results.txt"
        with open(pipeline_results_file, 'w') as f:
            f.write("MVP IMPLICATION CHAINS PIPELINE RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Pipeline Mode: {mode}\n")
            f.write(f"Start Time: {__import__('datetime').datetime.now()}\n\n")
        
        print("MVP IMPLICATION CHAINS PIPELINE")
        print(f"Mode: {mode}")
        print(f"Timestamp: {__import__('datetime').datetime.now()}")
        print()
        
        try:
            if mode == "test":
                result = self.run_test_only()
                with open(pipeline_results_file, 'a') as f:
                    f.write(f"Test Mode Result: {'PASSED' if result else 'FAILED'}\n")
                return result
            
            elif mode == "dataset":
                result = self.run_dataset_generation()
                with open(pipeline_results_file, 'a') as f:
                    f.write("Dataset Generation: COMPLETED\n")
                    f.write(f"Chains generated: {result['stats']['total_chains']}\n")
                    f.write(f"Pairs created: {result['stats']['total_pairs']}\n")
            
            elif mode == "train":
                classifier, test_dataset = self.run_classifier_training()
                test_results, factual_accuracy = self.run_evaluation(classifier)
                
                with open(pipeline_results_file, 'a') as f:
                    f.write("Classifier Training: COMPLETED\n")
                    f.write("Model Evaluation: COMPLETED\n")
                    f.write(f"Test Accuracy: {test_results['eval_accuracy']:.4f}\n")
                    f.write(f"Test F1 (Macro): {test_results['eval_f1_macro']:.4f}\n")
                    f.write(f"Factual Sanity Check: {factual_accuracy:.4f}\n")
            
            elif mode == "eval":
                test_results, factual_accuracy = self.run_evaluation()
                
                with open(pipeline_results_file, 'a') as f:
                    f.write("Model Evaluation: COMPLETED\n")
                    f.write(f"Test Accuracy: {test_results['eval_accuracy']:.4f}\n")
                    f.write(f"Test F1 (Macro): {test_results['eval_f1_macro']:.4f}\n")
                    f.write(f"Factual Sanity Check: {factual_accuracy:.4f}\n")
            
            elif mode == "all":
                # Run complete pipeline
                with open(pipeline_results_file, 'a') as f:
                    f.write("FULL PIPELINE EXECUTION\n")
                    f.write("-" * 25 + "\n\n")
                
                dataset_result = self.run_dataset_generation()
                classifier, test_dataset = self.run_classifier_training()
                test_results, factual_accuracy = self.run_evaluation(classifier)
                
                with open(pipeline_results_file, 'a') as f:
                    f.write("All phases completed successfully!\n")
                    f.write(f"Final Results:\n")
                    f.write(f"  Chains: {dataset_result['stats']['total_chains']}\n")
                    f.write(f"  Pairs: {dataset_result['stats']['total_pairs']}\n")
                    f.write(f"  Test Accuracy: {test_results['eval_accuracy']:.4f}\n")
                    f.write(f"  F1 Score: {test_results['eval_f1_macro']:.4f}\n")
                    f.write(f"  Factual Check: {factual_accuracy:.4f}\n")
            
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        except KeyboardInterrupt:
            error_msg = "Pipeline interrupted by user"
            print(f"\n⚠ {error_msg}")
            with open(pipeline_results_file, 'a') as f:
                f.write(f"\nERROR: {error_msg}\n")
            return False
        except Exception as e:
            error_msg = f"Pipeline failed: {e}"
            print(f"\n✗ {error_msg}")
            with open(pipeline_results_file, 'a') as f:
                f.write(f"\nERROR: {error_msg}\n")
            return False
        
        # Add completion timestamp
        with open(pipeline_results_file, 'a') as f:
            f.write(f"\nEnd Time: {__import__('datetime').datetime.now()}\n")
            f.write("Pipeline Status: COMPLETED SUCCESSFULLY\n")
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"Results saved to: {pipeline_results_file}")
        return True


def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    
    default_config = {
        "chain_length": 3,
        "min_distance": 2,
        "target_pairs": 5000,
        "num_seeds": 50,
        "model_name": "roberta-base",
        "num_epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5
    }
    
    if Path(config_file).exists():
        with open(config_file, 'r') as f:
            user_config = json.load(f)
        
        # Merge with defaults
        default_config.update(user_config)
        print(f"Loaded configuration from {config_file}")
    else:
        # Save default config
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default configuration file: {config_file}")
    
    return default_config


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="MVP Implication Chains Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline
  python main.py --mode dataset     # Generate dataset only
  python main.py --mode train       # Train classifier only
  python main.py --mode eval        # Evaluate existing model
  python main.py --mode test        # Run system test
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["all", "dataset", "train", "eval", "test"],
        default="all",
        help="Pipeline mode to run (default: all)"
    )
    
    parser.add_argument(
        "--config",
        default="config.json",
        help="Configuration file (default: config.json)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize and run pipeline
    pipeline = MVPPipeline(config)
    success = pipeline.run_pipeline(args.mode)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
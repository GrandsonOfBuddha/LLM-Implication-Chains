#!/usr/bin/env python3
"""
Results viewer for MVP Implication Chains project
View all generated text files in a structured format
"""

import os
import glob
from datetime import datetime


def view_all_results():
    """Display all result files in a structured format"""
    
    print("=" * 70)
    print("MVP IMPLICATION CHAINS - RESULTS VIEWER")
    print("=" * 70)
    print(f"Generated on: {datetime.now()}")
    print()
    
    # Define result files to look for
    result_files = {
        "pipeline_results.txt": "Pipeline Execution Results",
        "dataset_statistics.txt": "Dataset Generation Statistics", 
        "evaluation_results.txt": "Model Evaluation Results",
        "factual_sanity_check.txt": "Factual Sanity Check Results",
        "test_context_results.txt": "Context and Independence Test Results"
    }
    
    # Find all available result files
    available_files = []
    for filename in result_files.keys():
        if os.path.exists(filename):
            available_files.append((filename, result_files[filename]))
    
    if not available_files:
        print("No result files found. Please run the pipeline first.")
        return
    
    # Display each available file
    for filename, description in available_files:
        print(f"\n{'=' * 70}")
        print(f"{description.upper()}")
        print(f"File: {filename}")
        print(f"{'=' * 70}")
        
        try:
            with open(filename, 'r') as f:
                content = f.read()
                print(content)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
        
        print(f"{'=' * 70}")
    
    # Summary
    print(f"\nSUMMARY")
    print("-" * 20)
    print(f"Found {len(available_files)} result files:")
    for filename, description in available_files:
        file_size = os.path.getsize(filename) if os.path.exists(filename) else 0
        mod_time = datetime.fromtimestamp(os.path.getmtime(filename)) if os.path.exists(filename) else "Unknown"
        print(f"  {filename:<25} | {file_size:>6} bytes | {mod_time}")


def create_results_summary():
    """Create a consolidated summary of all results"""
    
    summary_file = "results_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("MVP IMPLICATION CHAINS - CONSOLIDATED RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {datetime.now()}\n\n")
        
        # Try to extract key metrics from various files
        metrics = {}
        
        # Extract from pipeline results
        if os.path.exists("pipeline_results.txt"):
            with open("pipeline_results.txt", 'r') as pf:
                content = pf.read()
                if "Test Accuracy:" in content:
                    for line in content.split('\n'):
                        if "Test Accuracy:" in line:
                            metrics['test_accuracy'] = line.split(':')[1].strip()
                        elif "F1 Score:" in line or "Test F1 (Macro):" in line:
                            metrics['f1_score'] = line.split(':')[1].strip()
                        elif "Factual Check:" in line:
                            metrics['factual_accuracy'] = line.split(':')[1].strip()
        
        # Extract from dataset statistics
        if os.path.exists("dataset_statistics.txt"):
            with open("dataset_statistics.txt", 'r') as ds:
                content = ds.read()
                for line in content.split('\n'):
                    if "Total pairs:" in line:
                        metrics['total_pairs'] = line.split(':')[1].strip()
                    elif "Total chains:" in line:
                        metrics['total_chains'] = line.split(':')[1].strip()
                    elif "Average chain length:" in line:
                        metrics['avg_chain_length'] = line.split(':')[1].strip()
        
        # Write summary
        f.write("KEY METRICS\n")
        f.write("-" * 15 + "\n")
        if metrics:
            for key, value in metrics.items():
                formatted_key = key.replace('_', ' ').title()
                f.write(f"{formatted_key:<20}: {value}\n")
        else:
            f.write("No metrics available. Please run the full pipeline.\n")
        
        f.write("\nFILES GENERATED\n")
        f.write("-" * 20 + "\n")
        
        # List all result files
        result_patterns = ["*.txt", "*.json", "*.jsonl"]
        all_files = []
        for pattern in result_patterns:
            all_files.extend(glob.glob(pattern))
        
        # Filter to result files only
        result_files = [f for f in all_files if any(keyword in f for keyword in 
                       ['results', 'statistics', 'evaluation', 'sanity', 'context', 'dataset', 'chains'])]
        
        for filename in sorted(result_files):
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                mod_time = datetime.fromtimestamp(os.path.getmtime(filename))
                f.write(f"{filename:<30} | {file_size:>8} bytes | {mod_time}\n")
        
        f.write(f"\nSummary created: {datetime.now()}\n")
    
    print(f"Consolidated summary created: {summary_file}")


if __name__ == "__main__":
    view_all_results()
    print("\n" + "=" * 70)
    create_results_summary()
    print("=" * 70)
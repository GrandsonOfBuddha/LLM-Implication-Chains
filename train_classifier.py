import os
import json
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from datasets import load_dataset
import pandas as pd


class NLIDataset(Dataset):
    """Dataset class for Natural Language Inference pairs"""
    
    def __init__(self, pairs: List[Dict], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        self.label_to_id = {"entails": 0, "contradicts": 1, "independent": 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Encode premise and hypothesis
        encoding = self.tokenizer(
            pair["premise"],
            pair["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.label_to_id[pair["label"]], dtype=torch.long)
        }


class NLIClassifier:
    """Natural Language Inference Classifier"""
    
    def __init__(self, model_name: str = "roberta-base", num_labels: int = 3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_data(self, dataset_file: str, 
                    train_split: float = 0.8, 
                    val_split: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and split the dataset"""
        
        # Load pairs from JSONL
        pairs = []
        with open(dataset_file, 'r') as f:
            for line in f:
                pairs.append(json.loads(line))
        
        # Shuffle and split
        import random
        random.shuffle(pairs)
        
        n = len(pairs)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        
        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:val_end]
        test_pairs = pairs[val_end:]
        
        # Create datasets
        train_dataset = NLIDataset(train_pairs, self.tokenizer)
        val_dataset = NLIDataset(val_pairs, self.tokenizer)
        test_dataset = NLIDataset(test_pairs, self.tokenizer)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro"),
            "f1_weighted": f1_score(labels, predictions, average="weighted")
        }
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
             output_dir: str = "./nli_model", 
             num_epochs: int = 3,
             batch_size: int = 16,
             learning_rate: float = 2e-5):
        """Train the NLI classifier"""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=2,
            report_to=None,  # Disable wandb/tensorboard
            learning_rate=learning_rate
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        
        return trainer
    
    def evaluate(self, test_dataset: Dataset, model_dir: str = "./nli_model"):
        """Evaluate the trained model"""
        
        # Load the trained model
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        trainer = Trainer(
            model=model,
            compute_metrics=self.compute_metrics
        )
        
        print("Evaluating model...")
        results = trainer.evaluate(test_dataset)
        
        # Get predictions for detailed analysis
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Create detailed classification report
        label_names = ["entails", "contradicts", "independent"]
        report = classification_report(
            true_labels, pred_labels, 
            target_names=label_names,
            output_dict=True
        )
        
        # Save evaluation results to text file
        with open("evaluation_results.txt", 'w') as f:
            f.write("MVP IMPLICATION CHAINS - MODEL EVALUATION RESULTS\n")
            f.write("=" * 55 + "\n")
            f.write(f"Evaluation Date: {__import__('datetime').datetime.now()}\n\n")
            
            f.write("OVERALL METRICS\n")
            f.write("-" * 20 + "\n")
            for key, value in results.items():
                if isinstance(value, float):
                    f.write(f"{key:<25}: {value:.4f}\n")
                else:
                    f.write(f"{key:<25}: {value}\n")
            
            f.write("\nCLASSIFICATION REPORT\n")
            f.write("-" * 25 + "\n")
            
            # Format classification report nicely
            f.write(f"{'Label':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
            f.write("-" * 60 + "\n")
            
            for label in label_names:
                if label in report:
                    metrics = report[label]
                    f.write(f"{label:<12} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} {metrics['support']:<10.0f}\n")
            
            f.write("-" * 60 + "\n")
            
            # Add macro and weighted averages
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in report:
                    metrics = report[avg_type]
                    f.write(f"{avg_type:<12} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} {metrics['support']:<10.0f}\n")
            
            f.write(f"\nAccuracy: {report['accuracy']:.4f}\n")
            
            # Add confusion matrix analysis
            try:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(true_labels, pred_labels)
                
                f.write("\nCONFUSION MATRIX\n")
                f.write("-" * 20 + "\n")
                f.write("Predicted ->   Entails  Contradicts  Independent\n")
                f.write("Actual\n")
                for i, label in enumerate(label_names):
                    f.write(f"{label:<12}  ")
                    for j in range(len(label_names)):
                        f.write(f"{cm[i][j]:>8}  ")
                    f.write("\n")
            except ImportError:
                f.write("\nConfusion matrix not available (sklearn import error)\n")
        
        print("Evaluation results saved to: evaluation_results.txt")
        
        return results, report
    
    def predict(self, premise: str, hypothesis: str, model_dir: str = "./nli_model"):
        """Make a prediction on a single premise-hypothesis pair"""
        
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Encode input
        encoding = tokenizer(
            premise, hypothesis,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert to labels
        label_names = ["entails", "contradicts", "independent"]
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
        
        return {
            "prediction": label_names[predicted_class],
            "confidence": confidence,
            "all_scores": {
                label: score.item() 
                for label, score in zip(label_names, predictions[0])
            }
        }


def run_factual_sanity_check(classifier: NLIClassifier, model_dir: str = "./nli_model"):
    """Run sanity checks on factual knowledge"""
    
    test_cases = [
        ("All dogs are animals", "My pet dog is an animal", "entails"),
        ("Birds can fly", "Penguins cannot fly", "contradicts"),
        ("Water freezes at 0°C", "Ice is frozen water", "entails"),
        ("All cats are mammals", "Some fish are colorful", "independent"),
        ("The Earth is round", "The Earth is flat", "contradicts"),
        ("Plants need sunlight", "Roses need sunlight", "entails"),
    ]
    
    print("\nFactual Sanity Check Results:")
    print("-" * 50)
    
    # Create results file
    results_file = "factual_sanity_check.txt"
    with open(results_file, 'w') as f:
        f.write("MVP IMPLICATION CHAINS - FACTUAL SANITY CHECK\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Date: {__import__('datetime').datetime.now()}\n\n")
        
        f.write("TEST CASES AND RESULTS\n")
        f.write("-" * 25 + "\n\n")
    
    correct = 0
    for i, (premise, hypothesis, expected) in enumerate(test_cases):
        result = classifier.predict(premise, hypothesis, model_dir)
        predicted = result["prediction"]
        confidence = result["confidence"]
        
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        # Console output
        print(f"Test Case {i+1}:")
        print(f"Premise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Expected: {expected} | Predicted: {predicted} | Confidence: {confidence:.3f}")
        print(f"Correct: {'✓' if is_correct else '✗'}")
        print()
        
        # File output
        with open(results_file, 'a') as f:
            f.write(f"Test Case {i+1}:\n")
            f.write(f"Premise: {premise}\n")
            f.write(f"Hypothesis: {hypothesis}\n")
            f.write(f"Expected: {expected}\n")
            f.write(f"Predicted: {predicted}\n")
            f.write(f"Confidence: {confidence:.3f}\n")
            f.write(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}\n")
            f.write(f"All Scores: {result['all_scores']}\n")
            f.write("\n" + "-" * 30 + "\n\n")
    
    accuracy = correct / len(test_cases)
    summary_line = f"Sanity Check Accuracy: {accuracy:.2%} ({correct}/{len(test_cases)})"
    print(summary_line)
    
    # Save summary to file
    with open(results_file, 'a') as f:
        f.write("SUMMARY\n")
        f.write("-" * 10 + "\n")
        f.write(f"Total test cases: {len(test_cases)}\n")
        f.write(f"Correct predictions: {correct}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        
        if accuracy >= 0.75:
            f.write("Result: PASSED (≥75% accuracy threshold)\n")
        else:
            f.write("Result: FAILED (<75% accuracy threshold)\n")
    
    print(f"Detailed results saved to: {results_file}")
    
    return accuracy


def main():
    """Main training and evaluation pipeline"""
    
    # Check if dataset exists
    dataset_file = "mvp_dataset.jsonl"
    if not os.path.exists(dataset_file):
        print(f"Dataset file {dataset_file} not found. Please run build_dataset.py first.")
        return
    
    # Initialize classifier
    classifier = NLIClassifier()
    
    # Prepare data
    train_dataset, val_dataset, test_dataset = classifier.prepare_data(dataset_file)
    
    # Train the model
    trainer = classifier.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=3,
        batch_size=16
    )
    
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
    factual_accuracy = run_factual_sanity_check(classifier)
    
    # Summary
    print("\n" + "="*50)
    print("MVP EVALUATION SUMMARY")
    print("="*50)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.3f}")
    print(f"Test F1 (Macro): {test_results['eval_f1_macro']:.3f}")
    print(f"Factual Sanity Check: {factual_accuracy:.3f}")
    print("="*50)


if __name__ == "__main__":
    main()
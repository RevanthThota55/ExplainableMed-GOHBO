"""
Hyperparameter optimization script using GOHBO algorithm.

Optimizes the learning rate for ResNet-18 on medical imaging datasets.
"""

import argparse
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config import get_config
from models.resnet18_medical import MedicalResNet18
from datasets.brain_tumor import create_brain_tumor_dataloaders
from datasets.chest_xray import create_chest_xray_dataloaders
from datasets.colorectal import create_colorectal_dataloaders
from training.optimizer import GOHBOOptimizer


def get_dataloaders(dataset_name: str, config: dict):
    """Get appropriate dataloaders for the specified dataset."""
    dataset_path = config['dataset']['data_path']
    batch_size = config['training']['batch_size']
    num_workers = 2  # Fewer workers for optimization
    image_size = config['dataset']['image_size']
    augmentation_config = config['augmentation']['train']

    if dataset_name == 'brain_tumor':
        return create_brain_tumor_dataloaders(
            dataset_path,
            batch_size,
            num_workers,
            image_size,
            augmentation_config
        )
    elif dataset_name == 'chest_xray':
        return create_chest_xray_dataloaders(
            dataset_path,
            batch_size,
            num_workers,
            image_size,
            augmentation_config
        )
    elif dataset_name == 'colorectal':
        return create_colorectal_dataloaders(
            dataset_path,
            batch_size,
            num_workers,
            image_size,
            augmentation_config
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description='Optimize hyperparameters using GOHBO')
    parser.add_argument('--dataset', type=str, default='brain_tumor',
                       choices=['brain_tumor', 'chest_xray', 'colorectal'],
                       help='Dataset to optimize for')
    parser.add_argument('--population_size', type=int, default=None,
                       help='GOHBO population size (overrides config)')
    parser.add_argument('--iterations', type=int, default=None,
                       help='GOHBO max iterations (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for optimization')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize optimization results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output during optimization')

    args = parser.parse_args()

    # Get configuration
    config = get_config(args.dataset)

    # Override GOHBO config with command line arguments
    if args.population_size:
        config['gohbo']['population_size'] = args.population_size
    if args.iterations:
        config['gohbo']['max_iterations'] = args.iterations

    print("\n" + "=" * 60)
    print(f"GOHBO HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Optimizing: Learning Rate")
    print(f"Population Size: {config['gohbo']['population_size']}")
    print(f"Max Iterations: {config['gohbo']['max_iterations']}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Create dataloaders
    print("\n Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(args.dataset, config)

    # Create optimizer
    print("\n Initializing GOHBO optimizer...")
    optimizer = GOHBOOptimizer(
        model_class=MedicalResNet18,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device
    )

    # Run optimization
    print("\n Starting optimization...")
    print(" This may take a while depending on your settings...")
    best_learning_rate = optimizer.optimize(verbose=args.verbose)

    # Get optimization summary
    summary = optimizer.get_optimization_summary()

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Best Learning Rate: {summary['best_learning_rate']:.6f}")
    print(f"Best Validation Loss: {summary['best_fitness']:.4f}")
    print(f"Total Evaluations: {summary['num_evaluations']}")

    print("\n Top 5 Learning Rates:")
    for i, lr_info in enumerate(summary['top_5_learning_rates'], 1):
        print(f"  {i}. LR: {lr_info['learning_rate']:.6f}, "
              f"Val Loss: {lr_info['val_loss']:.4f}, "
              f"Val Acc: {lr_info['val_acc']:.2f}%")

    print("\n Learning Rate Statistics:")
    lr_stats = summary['learning_rate_range']
    print(f"  Range: [{lr_stats['min']:.6f}, {lr_stats['max']:.6f}]")
    print(f"  Mean: {lr_stats['mean']:.6f}")
    print(f"  Std: {lr_stats['std']:.6f}")

    # Save results
    results_dir = config['paths']['results']
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': args.dataset,
        'best_learning_rate': best_learning_rate,
        'best_fitness': summary['best_fitness'],
        'optimization_summary': summary,
        'gohbo_config': config['gohbo'],
        'optimization_history': optimizer.optimization_history
    }

    results_path = results_dir / 'gohbo_optimization.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Optimization results saved to {results_path}")

    # Visualize if requested
    if args.visualize:
        plot_path = results_dir / 'gohbo_optimization_plot.png'
        optimizer.visualize_optimization(save_path=plot_path)

    # Print next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"1. Train the model with optimized learning rate:")
    print(f"   python train.py --dataset {args.dataset} --learning_rate optimized")
    print(f"\n2. Or specify the optimized learning rate directly:")
    print(f"   python train.py --dataset {args.dataset} --learning_rate {best_learning_rate:.6f}")
    print("=" * 60)

    return optimizer, summary


if __name__ == '__main__':
    optimizer, summary = main()
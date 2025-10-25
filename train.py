"""
Main training script for medical image classification with GOHBO-optimized ResNet-18.
"""

import argparse
import sys
from pathlib import Path
import torch
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config import get_config
from models.resnet18_medical import MedicalResNet18
from datasets.brain_tumor import create_brain_tumor_dataloaders
from datasets.chest_xray import create_chest_xray_dataloaders
from datasets.colorectal import create_colorectal_dataloaders
from training.trainer import Trainer


def get_dataloaders(dataset_name: str, config: dict):
    """
    Get appropriate dataloaders for the specified dataset.

    Args:
        dataset_name: Name of the dataset
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_path = config['dataset']['data_path']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
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
    parser = argparse.ArgumentParser(description='Train Medical Image Classification Model')
    parser.add_argument('--dataset', type=str, default='brain_tumor',
                       choices=['brain_tumor', 'chest_xray', 'colorectal'],
                       help='Dataset to train on')
    parser.add_argument('--learning_rate', type=str, default='optimized',
                       help='Learning rate (use "optimized" to load from optimization results)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--no_tensorboard', action='store_true',
                       help='Disable TensorBoard logging')

    args = parser.parse_args()

    # Get configuration
    config = get_config(args.dataset)

    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.no_tensorboard:
        config['logging']['tensorboard']['enabled'] = False

    print("\n" + "=" * 60)
    print(f"MEDICAL IMAGE CLASSIFICATION TRAINING")
    print("=" * 60)
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Classes: {config['dataset']['num_classes']}")
    print(f"Device: {args.device}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")

    # Get learning rate
    if args.learning_rate == 'optimized':
        # Try to load optimized learning rate
        optimization_results = config['paths']['results'] / 'gohbo_optimization.json'
        if optimization_results.exists():
            with open(optimization_results, 'r') as f:
                opt_results = json.load(f)
                learning_rate = opt_results['best_learning_rate']
                print(f"Using GOHBO-optimized learning rate: {learning_rate:.6f}")
        else:
            print("No optimization results found. Run optimize_hyperparams.py first.")
            print("Using default learning rate from config.")
            learning_rate = None
    else:
        learning_rate = float(args.learning_rate)
        print(f"Using specified learning rate: {learning_rate:.6f}")

    # Store learning rate in config
    if learning_rate:
        config['optimized_learning_rate'] = learning_rate

    # Create dataloaders
    print("\n Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(args.dataset, config)

    # Create model
    print("\n Creating model...")
    model = MedicalResNet18(
        num_classes=config['dataset']['num_classes'],
        input_channels=config['dataset']['channels'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone'],
        dropout_rate=config['model']['dropout_rate'],
        hidden_units=config['model']['hidden_units']
    )

    print(f"Model parameters: {model.get_num_trainable_params():,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train model
    print("\n Starting training...")
    print("=" * 60)

    history = trainer.train(
        num_epochs=config['training']['epochs'],
        learning_rate=learning_rate
    )

    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")

    # Save configuration with results
    results = {
        'dataset': args.dataset,
        'best_val_acc': trainer.best_val_acc,
        'best_val_loss': trainer.best_val_loss,
        'final_train_acc': history['train_acc'][-1],
        'final_train_loss': history['train_loss'][-1],
        'learning_rate': learning_rate if learning_rate else config['training']['optimizer'].get('lr', 1e-3),
        'epochs_trained': len(history['train_loss']),
        'config': config
    }

    results_path = config['paths']['results'] / f'training_results_{args.dataset}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")
    print(f"✓ Model checkpoints saved to {config['paths']['models'] / 'checkpoints'}")

    return trainer, history


if __name__ == '__main__':
    trainer, history = main()
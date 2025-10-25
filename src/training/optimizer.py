"""
GOHBO Optimizer wrapper for hyperparameter optimization.

Integrates GOHBO algorithm with the training pipeline to optimize learning rate.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, Optional, Callable, Any, Tuple
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.gohbo import GOHBO
from models.resnet18_medical import MedicalResNet18


class GOHBOOptimizer:
    """
    GOHBO optimizer wrapper for learning rate optimization.
    """

    def __init__(
        self,
        model_class: type,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        Initialize GOHBO optimizer wrapper.

        Args:
            model_class: Model class to instantiate
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device for training
        """
        self.model_class = model_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Create subset of data for faster optimization
        self.subset_train_loader = self._create_subset_loader(train_loader, 0.2)
        self.subset_val_loader = self._create_subset_loader(val_loader, 0.3)

        # GOHBO configuration
        self.gohbo_config = config['gohbo']
        self.lr_bounds = (
            self.gohbo_config['learning_rate_bounds']['min'],
            self.gohbo_config['learning_rate_bounds']['max']
        )

        # Results storage
        self.optimization_history = []
        self.best_learning_rate = None
        self.best_fitness = float('inf')

    def _create_subset_loader(self, loader: DataLoader, fraction: float) -> DataLoader:
        """
        Create a subset of the data loader for faster optimization.

        Args:
            loader: Original data loader
            fraction: Fraction of data to use

        Returns:
            Subset data loader
        """
        dataset = loader.dataset
        num_samples = len(dataset)
        num_subset = int(num_samples * fraction)

        # Random indices for subset
        indices = np.random.choice(num_samples, num_subset, replace=False)
        subset = Subset(dataset, indices)

        # Create new loader with subset
        subset_loader = DataLoader(
            subset,
            batch_size=loader.batch_size,
            shuffle=True,
            num_workers=0,  # Fewer workers for speed
            pin_memory=loader.pin_memory
        )

        return subset_loader

    def objective_function(self, learning_rate: float) -> float:
        """
        Objective function for GOHBO to minimize.

        Args:
            learning_rate: Learning rate to evaluate

        Returns:
            Validation loss (fitness value to minimize)
        """
        print(f"\n  Evaluating LR: {learning_rate:.6f}")

        # Create a fresh model instance
        model = self.model_class(
            num_classes=self.config['dataset']['num_classes'],
            input_channels=self.config['dataset']['channels'],
            pretrained=self.config['model']['pretrained'],
            dropout_rate=self.config['model']['dropout_rate'],
            hidden_units=self.config['model']['hidden_units']
        ).to(self.device)

        # Create optimizer with current learning rate
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.config['training']['optimizer']['weight_decay']
        )

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Quick training for a few epochs
        num_epochs = 5  # Quick evaluation
        model.train()

        for epoch in range(num_epochs):
            # Training loop on subset
            train_loss = 0.0
            for images, labels in self.subset_train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.subset_train_loader)

        # Validation to get fitness
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in self.subset_val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(self.subset_val_loader)
        val_acc = 100. * val_correct / val_total

        # Store result
        self.optimization_history.append({
            'learning_rate': learning_rate,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_loss': train_loss
        })

        print(f"    Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Return validation loss as fitness (to minimize)
        return val_loss

    def optimize(self, verbose: bool = True) -> float:
        """
        Run GOHBO optimization to find best learning rate.

        Args:
            verbose: Whether to print optimization progress

        Returns:
            Optimized learning rate
        """
        print("\n" + "=" * 60)
        print("GOHBO HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        print(f"Learning rate bounds: [{self.lr_bounds[0]:.1e}, {self.lr_bounds[1]:.1e}]")
        print(f"Population size: {self.gohbo_config['population_size']}")
        print(f"Max iterations: {self.gohbo_config['max_iterations']}")

        # Initialize GOHBO
        gohbo = GOHBO(
            objective_function=self.objective_function,
            bounds=self.lr_bounds,
            population_size=self.gohbo_config['population_size'],
            max_iterations=self.gohbo_config['max_iterations'],
            gwo_config=self.gohbo_config.get('gwo'),
            hbo_config=self.gohbo_config.get('hbo'),
            orthogonal_config=self.gohbo_config.get('orthogonal'),
            seed=self.gohbo_config.get('seed'),
            use_log_scale=self.gohbo_config['learning_rate_bounds'].get('scale') == 'log'
        )

        # Run optimization
        results = gohbo.optimize(verbose=verbose)

        # Extract best learning rate
        self.best_learning_rate = results['best_learning_rate']
        self.best_fitness = results['best_fitness']

        # Print results
        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"Best Learning Rate: {self.best_learning_rate:.6f}")
        print(f"Best Validation Loss: {self.best_fitness:.4f}")
        print(f"Iterations Completed: {results['iterations_completed']}")

        # Print convergence summary
        if results['convergence_history']:
            initial_fitness = results['convergence_history'][0]
            final_fitness = results['convergence_history'][-1]
            improvement = initial_fitness - final_fitness
            improvement_pct = (improvement / initial_fitness) * 100

            print(f"\nConvergence:")
            print(f"  Initial Fitness: {initial_fitness:.4f}")
            print(f"  Final Fitness: {final_fitness:.4f}")
            print(f"  Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")

        # Print component contributions
        print(f"\nComponent Best Solutions:")
        print(f"  GWO Alpha: LR = {10**results['gwo_alpha'][0]:.6f}" if self.gohbo_config['learning_rate_bounds'].get('scale') == 'log' else f"  GWO Alpha: LR = {results['gwo_alpha'][0]:.6f}")

        if results['hbo_heap']:
            best_hbo = results['hbo_heap'][0]
            hbo_lr = 10**best_hbo[0][0] if self.gohbo_config['learning_rate_bounds'].get('scale') == 'log' else best_hbo[0][0]
            print(f"  HBO Best: LR = {hbo_lr:.6f}")

        print("=" * 60)

        return self.best_learning_rate

    def get_optimization_summary(self) -> Dict:
        """
        Get summary of optimization results.

        Returns:
            Dictionary containing optimization summary
        """
        if not self.optimization_history:
            return {}

        # Sort history by validation loss
        sorted_history = sorted(
            self.optimization_history,
            key=lambda x: x['val_loss']
        )

        # Get top 5 learning rates
        top_5 = sorted_history[:5]

        summary = {
            'best_learning_rate': self.best_learning_rate,
            'best_fitness': self.best_fitness,
            'num_evaluations': len(self.optimization_history),
            'top_5_learning_rates': [
                {
                    'learning_rate': item['learning_rate'],
                    'val_loss': item['val_loss'],
                    'val_acc': item['val_acc']
                }
                for item in top_5
            ],
            'learning_rate_range': {
                'min': min(h['learning_rate'] for h in self.optimization_history),
                'max': max(h['learning_rate'] for h in self.optimization_history),
                'mean': np.mean([h['learning_rate'] for h in self.optimization_history]),
                'std': np.std([h['learning_rate'] for h in self.optimization_history])
            },
            'validation_loss_range': {
                'min': min(h['val_loss'] for h in self.optimization_history),
                'max': max(h['val_loss'] for h in self.optimization_history),
                'mean': np.mean([h['val_loss'] for h in self.optimization_history]),
                'std': np.std([h['val_loss'] for h in self.optimization_history])
            }
        }

        return summary

    def visualize_optimization(self, save_path: Optional[Path] = None):
        """
        Visualize the optimization process.

        Args:
            save_path: Optional path to save the plot
        """
        if not self.optimization_history:
            print("No optimization history to visualize")
            return

        import matplotlib.pyplot as plt

        # Extract data
        learning_rates = [h['learning_rate'] for h in self.optimization_history]
        val_losses = [h['val_loss'] for h in self.optimization_history]
        val_accs = [h['val_acc'] for h in self.optimization_history]

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Learning Rate vs Validation Loss
        axes[0].scatter(learning_rates, val_losses, alpha=0.6, c='blue')
        axes[0].scatter([self.best_learning_rate], [self.best_fitness],
                       color='red', s=100, marker='*', label='Best')
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('Learning Rate Optimization')
        axes[0].set_xscale('log')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot 2: Learning Rate vs Validation Accuracy
        axes[1].scatter(learning_rates, val_accs, alpha=0.6, c='green')
        best_acc = [h['val_acc'] for h in self.optimization_history
                   if h['learning_rate'] == self.best_learning_rate][0]
        axes[1].scatter([self.best_learning_rate], [best_acc],
                       color='red', s=100, marker='*', label='Best')
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Validation Accuracy (%)')
        axes[1].set_title('Learning Rate vs Accuracy')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.suptitle(f'GOHBO Optimization Results (Best LR: {self.best_learning_rate:.6f})')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Optimization plot saved to {save_path}")

        plt.show()
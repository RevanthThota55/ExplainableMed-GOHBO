"""
Colorectal Cancer Histopathology Dataset Loader

Handles loading and preprocessing of colorectal tissue microscopy images.
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Callable, Dict, List


class ColorectalDataset(Dataset):
    """
    Dataset class for Colorectal Cancer Histopathology Classification.

    8 tissue types:
    - ADI: Adipose
    - BACK: Background
    - DEB: Debris
    - LYM: Lymphocytes
    - MUC: Mucus
    - MUS: Smooth muscle
    - NORM: Normal colon mucosa
    - STR: Cancer-associated stroma
    - TUM: Colorectal adenocarcinoma epithelium
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Initialize Colorectal Dataset.

        Args:
            root_dir: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to apply to images
            target_transform: Optional transform to apply to labels
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Class names and mappings
        self.classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
        self.class_descriptions = {
            'ADI': 'Adipose tissue',
            'BACK': 'Background',
            'DEB': 'Debris',
            'LYM': 'Lymphocytes',
            'MUC': 'Mucus',
            'MUS': 'Smooth muscle',
            'NORM': 'Normal colon mucosa',
            'STR': 'Cancer-associated stroma',
            'TUM': 'Colorectal adenocarcinoma epithelium'
        }

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Load image paths and labels
        self.image_paths, self.labels = self._load_dataset()

        print(f"Loaded {len(self.image_paths)} colorectal tissue images for {split} split")

    def _load_dataset(self) -> Tuple[List[Path], List[int]]:
        """
        Load dataset image paths and labels.

        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []

        split_dir = self.root_dir / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for class_name in self.classes:
            class_dir = split_dir / class_name

            if class_dir.exists():
                # Get all image files (colorectal dataset may use .tif format)
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                    for img_path in class_dir.glob(ext):
                        image_paths.append(img_path)
                        labels.append(self.class_to_idx[class_name])

        return image_paths, labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.

        Returns:
            Dictionary mapping class names to counts
        """
        distribution = {cls: 0 for cls in self.classes}

        for label in self.labels:
            class_name = self.idx_to_class[label]
            distribution[class_name] += 1

        return distribution

    def get_sample_weights(self) -> torch.Tensor:
        """
        Calculate sample weights for balanced sampling.

        Returns:
            Tensor of sample weights
        """
        class_counts = torch.zeros(len(self.classes))

        for label in self.labels:
            class_counts[label] += 1

        # Calculate weights (inverse of class frequency)
        class_weights = 1.0 / class_counts
        sample_weights = torch.zeros(len(self.labels))

        for idx, label in enumerate(self.labels):
            sample_weights[idx] = class_weights[label]

        return sample_weights

    def get_cancer_vs_normal_distribution(self) -> Dict[str, int]:
        """
        Get distribution of cancer-related vs normal tissues.

        Returns:
            Dictionary with cancer and normal counts
        """
        cancer_classes = ['TUM', 'STR', 'DEB']  # Cancer-related
        normal_classes = ['NORM', 'MUS', 'ADI']  # Normal tissues
        immune_classes = ['LYM']  # Immune cells
        other_classes = ['BACK', 'MUC']  # Other

        distribution = {
            'cancer_related': 0,
            'normal': 0,
            'immune': 0,
            'other': 0
        }

        for label in self.labels:
            class_name = self.idx_to_class[label]

            if class_name in cancer_classes:
                distribution['cancer_related'] += 1
            elif class_name in normal_classes:
                distribution['normal'] += 1
            elif class_name in immune_classes:
                distribution['immune'] += 1
            else:
                distribution['other'] += 1

        return distribution


def get_colorectal_transforms(
    image_size: Tuple[int, int] = (224, 224),
    split: str = 'train',
    augmentation_config: Optional[Dict] = None
) -> transforms.Compose:
    """
    Get appropriate transforms for colorectal dataset.

    Args:
        image_size: Target image size
        split: Dataset split ('train', 'val', 'test')
        augmentation_config: Augmentation configuration dictionary

    Returns:
        Composition of transforms
    """
    # ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train' and augmentation_config:
        # Training transforms with augmentation
        # Histopathology images can benefit from various augmentations
        transform_list = [
            transforms.Resize((image_size[0] + 20, image_size[1] + 20)),
            transforms.RandomCrop(image_size)
        ]

        # Add augmentations based on config
        if augmentation_config.get('random_horizontal_flip', {}).get('enabled', True):
            transform_list.append(
                transforms.RandomHorizontalFlip(
                    p=augmentation_config['random_horizontal_flip'].get('p', 0.5)
                )
            )

        # Vertical flip is OK for microscopy images
        if augmentation_config.get('random_vertical_flip', {}).get('enabled', True):
            transform_list.append(
                transforms.RandomVerticalFlip(
                    p=augmentation_config['random_vertical_flip'].get('p', 0.5)
                )
            )

        # Rotation is common for microscopy images
        if augmentation_config.get('random_rotation', {}).get('enabled', True):
            transform_list.append(
                transforms.RandomRotation(
                    degrees=augmentation_config['random_rotation'].get('degrees', 90)
                )
            )

        # Color variations are important for histopathology
        if augmentation_config.get('color_jitter', {}).get('enabled', True):
            config = augmentation_config['color_jitter']
            transform_list.append(
                transforms.ColorJitter(
                    brightness=config.get('brightness', 0.3),
                    contrast=config.get('contrast', 0.3),
                    saturation=config.get('saturation', 0.3),
                    hue=config.get('hue', 0.15)
                )
            )

        if augmentation_config.get('random_affine', {}).get('enabled', True):
            config = augmentation_config['random_affine']
            transform_list.append(
                transforms.RandomAffine(
                    degrees=config.get('degrees', 15),
                    translate=config.get('translate', (0.1, 0.1)),
                    scale=config.get('scale', (0.8, 1.2)),
                    shear=config.get('shear', 10)
                )
            )

        # Elastic deformation (common in histopathology augmentation)
        if augmentation_config.get('gaussian_blur', {}).get('enabled', True):
            transform_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(
                        kernel_size=augmentation_config['gaussian_blur'].get('kernel_size', 5)
                    )],
                    p=augmentation_config['gaussian_blur'].get('p', 0.2)
                )
            )

        transform_list.extend([
            transforms.ToTensor(),
            normalize
        ])

        if augmentation_config.get('random_erasing', {}).get('enabled', True):
            config = augmentation_config['random_erasing']
            transform_list.append(
                transforms.RandomErasing(
                    p=config.get('p', 0.2),
                    scale=config.get('scale', (0.02, 0.2)),
                    ratio=config.get('ratio', (0.3, 3.3))
                )
            )

    else:
        # Validation/Test transforms (no augmentation)
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize
        ]

    return transforms.Compose(transform_list)


def create_colorectal_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[Dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for colorectal dataset.

    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        image_size: Target image size
        augmentation_config: Augmentation configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create transforms
    train_transform = get_colorectal_transforms(
        image_size, 'train', augmentation_config
    )
    val_transform = get_colorectal_transforms(image_size, 'val')
    test_transform = get_colorectal_transforms(image_size, 'test')

    # Create datasets
    train_dataset = ColorectalDataset(data_dir, 'train', train_transform)
    val_dataset = ColorectalDataset(data_dir, 'val', val_transform)
    test_dataset = ColorectalDataset(data_dir, 'test', test_transform)

    # Get sample weights for balanced sampling
    sample_weights = train_dataset.get_sample_weights()
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Print dataset statistics
    print("\nColorectal Histopathology Dataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print("\nClass distribution (train):")
    for cls, count in train_dataset.get_class_distribution().items():
        desc = train_dataset.class_descriptions[cls]
        print(f"  {cls:5s} ({desc:35s}): {count:5d}")

    print("\nTissue type distribution (train):")
    tissue_dist = train_dataset.get_cancer_vs_normal_distribution()
    for tissue_type, count in tissue_dist.items():
        print(f"  {tissue_type:15s}: {count:5d}")

    return train_loader, val_loader, test_loader
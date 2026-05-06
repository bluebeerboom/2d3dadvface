"""
Example usage of CFP Dataset Loader
"""

import torch
from cfp_dataset_loader import CFPLoader, create_cfp_dataloaders


def main():
    # Path to your CFP dataset
    root_dir = r"d:\collegedata\2026\研二\警学项目申报\2d3dadvface\cfp-dataset"
    
    print("=" * 60)
    print("CFP Dataset Loader - Example Usage")
    print("=" * 60)
    
    # Method 1: Using create_cfp_dataloaders (recommended)
    print("\n1. Using create_cfp_dataloaders:")
    train_loader, test_loader = create_cfp_dataloaders(
        root_dir=root_dir,
        batch_size=16,
        mode='frontal',  # Options: 'frontal', 'profile', 'both'
        split_ratio=0.8,
        num_workers=0
    )
    
    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    # Iterate through one batch
    for images, labels in train_loader:
        print(f"\nImage batch shape: {images.shape}")  # [batch_size, 3, 112, 112]
        print(f"Label batch shape: {labels.shape}")    # [batch_size]
        print(f"Sample labels: {labels[:5].tolist()}")
        break
    
    # Method 2: Using CFPLoader directly
    print("\n\n2. Using CFPLoader directly:")
    dataset = CFPLoader(
        root_dir=root_dir,
        mode='both',
        train=True,
        split_ratio=0.8
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a single sample
    image, label = dataset[0]
    print(f"Single image shape: {image.shape}")  # [3, 112, 112]
    print(f"Single label: {label}")
    
    # Create DataLoader from dataset
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    print(f"\nNumber of batches: {len(loader)}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

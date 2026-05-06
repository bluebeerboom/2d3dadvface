"""
Quick test to verify ArcFace training pipeline works correctly
"""

import torch
import torch.nn as nn
from cfp_dataset_loader import create_cfp_dataloaders
from arcface_model import ArcFaceResNet50


def test_training_pipeline():
    """Test that training pipeline works end-to-end"""
    
    print("=" * 80)
    print("Testing ArcFace Training Pipeline")
    print("=" * 80)
    
    # Configuration
    ROOT_DIR = r"d:\collegedata\2026\研二\警学项目申报\2d3dadvface\cfp-dataset"
    BATCH_SIZE = 16  # Increased to avoid BatchNorm issues with small batches
    NUM_EPOCHS = 3  # Just test a few epochs
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create data loaders
    print("\n[1/4] Loading CFP dataset...")
    train_loader, val_loader = create_cfp_dataloaders(
        root_dir=ROOT_DIR,
        batch_size=BATCH_SIZE,
        mode='frontal',
        split_ratio=0.8,
        num_workers=0
    )
    
    num_classes = len(set(train_loader.dataset.labels))
    # Ensure num_classes accounts for all possible label values
    actual_num_classes = max(train_loader.dataset.labels) + 1
    print(f"  - Number of classes: {actual_num_classes} (unique: {num_classes})")
    print(f"  - Training samples: {len(train_loader.dataset)}")
    print(f"  - Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\n[2/4] Creating ArcFace ResNet50 model...")
    model = ArcFaceResNet50(
        num_classes=actual_num_classes,
        embedding_dim=512,
        pretrained=False  # Use random init for testing
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")
    
    # Setup optimizer and loss
    print("\n[3/4] Setting up optimizer...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Test training loop
    print("\n[4/4] Testing training loop...")
    print("-" * 80)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, labels)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {accuracy:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Update learning rate
        scheduler.step()
    
    print("-" * 80)
    
    # Test inference
    print("\n[5/5] Testing inference...")
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(val_loader))
        images = images.to(device)
        
        # Get embeddings
        embeddings = model.get_embedding(images)
        
        print(f"  - Input shape: {images.shape}")
        print(f"  - Embedding shape: {embeddings.shape}")
        print(f"  - Embedding norms: {torch.norm(embeddings, dim=1)}")
        print(f"  - Labels: {labels.tolist()}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)
    print("\nTraining pipeline is working correctly.")
    print("You can now run full training with: python train_arcface.py")
    

if __name__ == "__main__":
    test_training_pipeline()

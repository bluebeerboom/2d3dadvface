"""
Train Face Recognition Model and Build Gallery
训练人脸识别模型并建立特征库

Workflow:
1. Train ArcFace model on cfp-dataset
2. Build gallery features from cfp-dataset
3. Save model checkpoint and gallery features
"""

import os
import sys
import argparse
import torch
from torchvision import transforms
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.cfp_dataset_loader import CFPLoader
from models.arcface_model import ArcFaceResNet50
from torch.utils.data import DataLoader


def train_model(root_dir, epochs=30, batch_size=32, lr=0.01, save_dir='../../models', device=None):
    """
    Train face recognition model
    
    Args:
        root_dir: cfp-dataset root directory
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        save_dir: directory to save model checkpoints
        device: computing device
    
    Returns:
        model: trained model
        actual_num_classes: number of classes in training data
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Phase 1: Training Face Recognition Model")
    print("="*80)
    
    # Create data loader
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = CFPLoader(
        root_dir=root_dir,
        mode='frontal',
        transform=train_transform,
        train=True,
        split_ratio=1.0  # Use all data
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Get actual number of classes
    actual_num_classes = len(set(train_loader.dataset.labels))
    print(f"Actual number of classes in training data: {actual_num_classes}")
    
    # Create model
    embedding_dim = 512
    model = ArcFaceResNet50(
        num_classes=actual_num_classes,
        embedding_dim=embedding_dim,
        pretrained=True
    ).to(device)
    print(f"Model created with {actual_num_classes} classes")
    
    # Setup optimizer and loss
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        import time
        start_time = time.time()
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through ArcFace
            features = model.backbone(images)
            features = model.global_pool(features)
            embeddings = model.bottleneck(features)
            logits = model.arcface(embeddings, labels)
            
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            avg_loss = total_loss / (len(progress_bar))
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        scheduler.step()
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Time: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}%")
        
        # Save model every 5 epochs and last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'num_classes': actual_num_classes,
                'embedding_dim': embedding_dim,
            }, checkpoint_path)
            print(f"  ✓ Model saved to {checkpoint_path} (Train Acc: {train_acc:.2f}%)")
        
        print("-" * 80)
    
    print(f"\n✓ Training completed! Final Train Acc: {train_acc:.2f}%")
    return model, actual_num_classes, embedding_dim


def build_gallery(model, root_dir, save_path='gallery_features.pth', device=None):
    """
    Build gallery features from cfp-dataset
    
    Args:
        model: trained model
        root_dir: cfp-dataset root directory
        save_path: path to save gallery features
        device: computing device
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("Phase 2: Building Gallery Features")
    print("="*80)
    
    model.eval()
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = CFPLoader(
        root_dir=root_dir,
        mode='both',  # Use both frontal and profile
        transform=transform,
        train=True,
        split_ratio=1.0  # Use all data
    )
    
    # Extract features for each person
    gallery_features = {}  # {person_id: embedding}
    
    unique_labels = sorted(set(dataset.labels))
    print(f"Total persons in gallery: {len(unique_labels)}")
    
    # Create reverse mapping: mapped_label -> original_person_id
    reverse_mapping = {v: k for k, v in dataset.label_mapping.items()}
    print(f"Label mapping: {len(dataset.label_mapping)} persons")
    
    for mapped_label in tqdm(unique_labels, desc="Extracting features"):
        # Convert mapped label back to original person ID
        original_person_id = reverse_mapping[mapped_label]
        
        # Get all image indices for this person
        indices = [i for i, label in enumerate(dataset.labels) if label == mapped_label]
        
        embeddings = []
        for idx in indices:
            image, _ = dataset[idx]
            image = image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = model.get_embedding(image)
                # Remove batch dimension to get [512]
                embeddings.append(emb.squeeze(0).cpu())
        
        # Take average as this person's feature
        if embeddings:
            avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
            # Store with original person_id (not mapped label)
            gallery_features[original_person_id] = avg_embedding
    
    # Save gallery
    torch.save(gallery_features, save_path)
    print(f"\n✓ Gallery features saved to {save_path}")
    print(f"  Total persons: {len(gallery_features)}")
    print(f"  Embedding dimension: {list(gallery_features.values())[0].shape[0]}")
    
    return gallery_features


def main():
    parser = argparse.ArgumentParser(description='Train Face Recognition Model and Build Gallery')
    parser.add_argument('--cfp-root', type=str, required=True, help='Path to cfp-dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='../../models', help='Directory to save model')
    parser.add_argument('--gallery-path', type=str, default='../../galleries/gallery_features.pth', help='Path to save gallery')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Phase 1: Train model
    model, num_classes, embedding_dim = train_model(
        root_dir=args.cfp_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device=device
    )
    
    # Phase 2: Build gallery
    build_gallery(
        model=model,
        root_dir=args.cfp_root,
        save_path=args.gallery_path,
        device=device
    )
    
    print("\n" + "="*80)
    print("TRAINING AND GALLERY BUILDING COMPLETED!")
    print("="*80)
    print(f"Model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")
    print(f"Gallery saved to: {args.gallery_path}")
    print(f"Classes: {num_classes}, Embedding dim: {embedding_dim}")
    print("="*80)
    print("\nNow you can run test_recognition.py to evaluate on occluded dataset")


if __name__ == "__main__":
    main()

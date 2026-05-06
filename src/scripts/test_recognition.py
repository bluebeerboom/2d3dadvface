"""
Test Face Recognition on Occluded Dataset
在遮挡数据集上测试人脸识别准确率

Workflow:
1. Load trained model from checkpoint
2. Load gallery features
3. Test recognition accuracy on occluded dataset (or custom dataset)
"""

import os
import sys
import argparse
import torch
from torchvision import transforms
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.occluded_loader import create_occluded_dataloader
from models.arcface_model import ArcFaceResNet50


def test_recognition(model_path, gallery_path, occluded_root, mode='both', device=None):
    """
    Test face recognition accuracy
    
    Args:
        model_path: path to trained model checkpoint
        gallery_path: path to gallery features
        occluded_root: path to occluded dataset
        mode: 'frontal', 'profile', or 'both'
        device: computing device
    
    Returns:
        accuracy: recognition accuracy
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("Testing Face Recognition on Occluded Dataset")
    print("="*80)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to get num_classes and embedding_dim from checkpoint
    # For backward compatibility with old checkpoints
    if 'num_classes' in checkpoint and 'embedding_dim' in checkpoint:
        num_classes = checkpoint['num_classes']
        embedding_dim = checkpoint['embedding_dim']
        print(f"Loading model config from checkpoint: {num_classes} classes, {embedding_dim} dim")
    else:
        # Infer from model weights (for old checkpoints)
        arcface_weight = checkpoint['model_state_dict']['arcface.weight']
        num_classes = arcface_weight.shape[0]
        embedding_dim = arcface_weight.shape[1]
        print(f"Inferred model config from weights: {num_classes} classes, {embedding_dim} dim")
    
    print(f"Loading model from {model_path}")
    
    model = ArcFaceResNet50(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded (epoch {checkpoint['epoch']}, train acc: {checkpoint['train_acc']:.2f}%)")
    
    # Load gallery
    gallery_features = torch.load(gallery_path, map_location=device)
    print(f"Gallery loaded: {len(gallery_features)} persons")
    
    # Load occluded dataset
    occluded_loader = create_occluded_dataloader(
        root_dir=occluded_root,
        batch_size=32,
        mode=mode,
        num_workers=0
    )
    
    print(f"Testing on occluded dataset (mode={mode})")
    print(f"Total test images: {len(occluded_loader.dataset)}")
    
    # Pre-compute gallery tensor for fast similarity computation
    gallery_ids = sorted(gallery_features.keys())
    print(f"Gallery contains {len(gallery_ids)} persons")
    
    # Ensure all embeddings are tensors with correct dimensions
    gallery_embeddings_list = []
    for pid in gallery_ids:
        emb = gallery_features[pid]
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        # Ensure embedding is 1D [512], remove any extra dimensions
        emb = emb.squeeze()
        if emb.dim() == 0:
            emb = emb.unsqueeze(0)
        gallery_embeddings_list.append(emb)
    
    gallery_tensor = torch.stack(gallery_embeddings_list).to(device)
    print(f"Gallery tensor shape: {gallery_tensor.shape}")
    
    # Testing
    correct = 0
    total = 0
    
    progress_bar = tqdm(occluded_loader, desc="Testing")
    
    with torch.no_grad():
        for images, true_labels in progress_bar:
            images = images.to(device)
            
            # Batch extract embeddings
            embeddings = model.get_embedding(images)  # [batch_size, 512]
            
            # Batch compute similarities
            # embeddings: [batch_size, 512]
            # gallery_tensor: [num_persons, 512]
            similarities = torch.nn.functional.cosine_similarity(
                embeddings.unsqueeze(1),  # [batch_size, 1, 512]
                gallery_tensor.unsqueeze(0),  # [1, num_persons, 512]
                dim=2  # [batch_size, num_persons]
            )
            
            # Find most similar person
            _, predicted_indices = similarities.max(dim=1)
            predicted_ids = [gallery_ids[idx.item()] for idx in predicted_indices]
            
            # Count correct predictions
            for pred_id, true_id in zip(predicted_ids, true_labels.tolist()):
                # Only count persons that exist in gallery
                if true_id in gallery_features:
                    total += 1
                    if pred_id == true_id:
                        correct += 1
            
            acc = 100. * correct / total if total > 0 else 0
            progress_bar.set_postfix({'acc': f'{acc:.2f}%'})
    
    accuracy = 100. * correct / total if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Recognition Results:")
    print(f"  Mode: {mode}")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Top-1 Accuracy: {accuracy:.2f}%")
    print(f"{'='*80}")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Test Face Recognition Accuracy')
    parser.add_argument('--model-path', type=str, default='../../models/best_model.pth', 
                       help='Path to trained model checkpoint')
    parser.add_argument('--gallery-path', type=str, default='../../galleries/gallery_features.pth', 
                       help='Path to gallery features')
    parser.add_argument('--occluded-root', type=str, required=True, 
                       help='Path to occluded dataset')
    parser.add_argument('--mode', type=str, default='both', choices=['frontal', 'profile', 'both'],
                       help='Test mode: frontal, profile, or both')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Verify files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.gallery_path):
        print(f"Error: Gallery file not found: {args.gallery_path}")
        sys.exit(1)
    
    if not os.path.exists(args.occluded_root):
        print(f"Error: Occluded dataset not found: {args.occluded_root}")
        sys.exit(1)
    
    # Test recognition
    accuracy = test_recognition(
        model_path=args.model_path,
        gallery_path=args.gallery_path,
        occluded_root=args.occluded_root,
        mode=args.mode,
        device=device
    )
    
    print(f"\n✓ Final Accuracy ({args.mode}): {accuracy:.2f}%")


if __name__ == "__main__":
    main()

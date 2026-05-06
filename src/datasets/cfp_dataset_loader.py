import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob


class CFPLoader(Dataset):
    """
    PyTorch Dataset for loading CFP (Celebrities in Frontal-Profile) dataset
    
    Structure:
        Data/Images/{person_id}/frontal/*.jpg  (10 images per person)
        Data/Images/{person_id}/profile/*.jpg   (4 images per person)
    
    Args:
        root_dir (str): Root directory of the CFP dataset
        mode (str): 'frontal', 'profile', or 'both' - which type of images to load
        transform (callable, optional): Optional transform to be applied on a sample
        train (bool): If True, use training split; if False, use test split
        split_ratio (float): Ratio of data to use for training (default: 0.8)
        seed (int): Random seed for reproducible splits (default: 42)
    """
    
    def __init__(self, root_dir, mode='frontal', transform=None, 
                 train=True, split_ratio=0.8, seed=42):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # Set default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        # Get all person IDs
        self.person_ids = sorted([
            d for d in os.listdir(os.path.join(root_dir, 'Data', 'Images'))
            if os.path.isdir(os.path.join(root_dir, 'Data', 'Images', d))
        ])
        
        # Split persons into train/test
        torch.manual_seed(seed)
        indices = torch.randperm(len(self.person_ids))
        split_idx = int(len(self.person_ids) * split_ratio)
        
        if train:
            self.selected_persons = [self.person_ids[i] for i in indices[:split_idx]]
        else:
            self.selected_persons = [self.person_ids[i] for i in indices[split_idx:]]
        
        # Build image-label pairs
        self.image_paths = []
        self.labels = []
        self.label_mapping = {}  # Map original person_id to contiguous class index
        
        self._build_dataset()
        
        print(f"CFP Dataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Split: {'train' if train else 'test'}")
        print(f"  Persons: {len(self.selected_persons)}")
        print(f"  Total images: {len(self.image_paths)}")
    
    def _build_dataset(self):
        """Build list of image paths and corresponding labels"""
        # First pass: collect all images and their original person IDs
        temp_images = []
        temp_labels_original = []
        
        for person_id in self.selected_persons:
            original_label = int(person_id)
            
            if self.mode in ['frontal', 'both']:
                frontal_path = os.path.join(
                    self.root_dir, 'Data', 'Images', person_id, 'frontal'
                )
                if os.path.exists(frontal_path):
                    frontal_images = sorted(glob.glob(os.path.join(frontal_path, '*.jpg')))
                    for img_path in frontal_images:
                        temp_images.append(img_path)
                        temp_labels_original.append(original_label)
            
            if self.mode in ['profile', 'both']:
                profile_path = os.path.join(
                    self.root_dir, 'Data', 'Images', person_id, 'profile'
                )
                if os.path.exists(profile_path):
                    profile_images = sorted(glob.glob(os.path.join(profile_path, '*.jpg')))
                    for img_path in profile_images:
                        temp_images.append(img_path)
                        temp_labels_original.append(original_label)
        
        # Second pass: create label mapping ONLY for persons with actual images
        unique_persons_with_images = sorted(set(temp_labels_original))
        for idx, person_id in enumerate(unique_persons_with_images):
            self.label_mapping[person_id] = idx
        
        # Third pass: map labels to contiguous indices
        for img_path, original_label in zip(temp_images, temp_labels_original):
            self.image_paths.append(img_path)
            self.labels.append(self.label_mapping[original_label])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label


def create_cfp_dataloaders(root_dir, batch_size=32, mode='frontal', 
                          split_ratio=0.8, num_workers=4):
    """
    Create train and test DataLoaders for CFP dataset
    
    Args:
        root_dir (str): Root directory of the CFP dataset
        batch_size (int): Batch size for DataLoader
        mode (str): 'frontal', 'profile', or 'both'
        split_ratio (float): Ratio of data to use for training
        num_workers (int): Number of workers for DataLoader
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # 新增：随机旋转
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 增强
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),  # 新增：随机擦除（必须在ToTensor之后）
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    train_dataset = CFPLoader(
        root_dir=root_dir,
        mode=mode,
        transform=train_transform,
        train=True,
        split_ratio=split_ratio
    )
    
    test_dataset = CFPLoader(
        root_dir=root_dir,
        mode=mode,
        transform=test_transform,
        train=False,
        split_ratio=split_ratio
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # Keep all samples for testing
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    root_dir = r"d:\collegedata\2026\研二\警学项目申报\2d3dadvface\cfp-dataset"
    
    # Test with frontal images only
    print("Testing frontal mode...")
    train_loader, test_loader = create_cfp_dataloaders(
        root_dir=root_dir,
        batch_size=8,
        mode='frontal',
        num_workers=0  # Set to 0 to avoid multiprocessing issues in Windows
    )
    
    # Check one batch from training loader
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i+1} shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample labels: {labels}")
        if i >= 2:  # Only show first 3 batches
            break
    
    # Test with profile images only
    print("\nTesting profile mode...")
    train_loader, test_loader = create_cfp_dataloaders(
        root_dir=root_dir,
        batch_size=8,
        mode='profile',
        num_workers=0
    )
    
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i+1} shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample labels: {labels}")
        if i >= 2:
            break
    
    # Test with both frontal and profile images
    print("\nTesting both mode...")
    train_loader, test_loader = create_cfp_dataloaders(
        root_dir=root_dir,
        batch_size=8,
        mode='both',
        num_workers=0
    )
    
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i+1} shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample labels: {labels}")
        if i >= 2:
            break
    
    print("\nAll tests completed successfully!")

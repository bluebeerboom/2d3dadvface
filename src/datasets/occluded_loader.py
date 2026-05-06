"""
Occluded Dataset Loader for Face Recognition
加载遮挡人脸数据用于识别测试
"""

import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OccludedLoader(Dataset):
    """
    加载 occluded 遮挡人脸数据集
    
    数据结构:
        occluded/
        ├── 001/
        │   ├── frontal/*.jpg
        │   └── profile/*.jpg
        ├── 002/
        │   ├── frontal/*.jpg
        │   └── profile/*.jpg
        ...
    
    Returns:
        (image_tensor, person_id)
    """
    
    def __init__(self, root_dir, mode='both', transform=None):
        """
        Args:
            root_dir: occluded数据集根目录
            mode: 'frontal', 'profile', or 'both'
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        self._build_dataset()
        
        print(f"Occluded Dataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Total persons: {len(set(self.labels))}")
        print(f"  Total images: {len(self.image_paths)}")
    
    def _build_dataset(self):
        """构建图像路径和标签列表"""
        # 遍历所有人员文件夹
        for person_id in sorted(os.listdir(self.root_dir)):
            person_dir = os.path.join(self.root_dir, person_id)
            
            if not os.path.isdir(person_dir):
                continue
            
            try:
                pid = int(person_id)
            except ValueError:
                continue
            
            # 根据mode选择视图
            views = []
            if self.mode == 'frontal':
                views = ['frontal']
            elif self.mode == 'profile':
                views = ['profile']
            else:  # both
                views = ['frontal', 'profile']
            
            # 加载指定视图的图像
            for view in views:
                view_dir = os.path.join(person_dir, view)
                if os.path.exists(view_dir):
                    for img_file in sorted(glob.glob(os.path.join(view_dir, '*.jpg'))):
                        self.image_paths.append(img_file)
                        self.labels.append(pid)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor [3, 112, 112]
            label: int (person_id)
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label
    
    def get_person_images(self, person_id):
        """获取某个人的所有图像路径"""
        person_images = []
        for path, label in zip(self.image_paths, self.labels):
            if label == person_id:
                person_images.append(path)
        return person_images


def create_occluded_dataloader(root_dir, batch_size=32, mode='both', num_workers=0):
    """
    创建 occluded 数据集的 DataLoader
    
    Args:
        root_dir: occluded数据集根目录
        batch_size: batch大小
        mode: 'frontal', 'profile', or 'both'
        num_workers: 数据加载线程数
    
    Returns:
        DataLoader
    """
    # 测试时不需要数据增强，只做基本的resize和normalize
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = OccludedLoader(
        root_dir=root_dir,
        mode=mode,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试时不需要shuffle
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试代码
    ROOT_DIR = r"d:\collegedata\2026\研二\警学项目申报\2d3dadvface\occluded"
    
    print("=" * 80)
    print("Testing OccludedLoader...")
    print("=" * 80)
    
    # 测试 both mode
    loader = create_occluded_dataloader(ROOT_DIR, batch_size=4, mode='both')
    
    print("\nLoading sample batch...")
    images, labels = next(iter(loader))
    print(f"Image shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Unique persons in batch: {len(set(labels.tolist()))}")
    
    print("\n✓ OccludedLoader test passed!")

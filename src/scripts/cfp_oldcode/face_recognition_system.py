"""
Face Recognition System for Access Control
门禁人脸识别系统

工作流程:
1. 使用 cfp-dataset 训练 ArcFace 模型
2. 从 cfp-dataset 建立特征库（500人的embedding）
3. 在 occluded 数据集上测试识别准确率
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

from cfp_dataset_loader import create_cfp_dataloaders
from occluded_loader import create_occluded_dataloader
from arcface_model import ArcFaceResNet50


def cosine_similarity(embedding1, embedding2):
    """计算两个embedding的余弦相似度"""
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)


class FaceRecognitionSystem:
    """人脸识别系统"""
    
    def __init__(self, num_classes=500, embedding_dim=512, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = embedding_dim
        
        # 创建模型
        self.model = ArcFaceResNet50(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=True
        ).to(self.device)
        
        print(f"Model created on {self.device}")
        print(f"Embedding dimension: {embedding_dim}")
    
    def train(self, root_dir, epochs=30, batch_size=32, lr=0.001, save_dir='../../models'):
        """
        使用 cfp-dataset 训练模型
        
        Args:
            root_dir: cfp-dataset根目录
            epochs: 训练轮数
            batch_size: batch大小
            lr: 学习率
            save_dir: 模型保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("Phase 1: Training Face Recognition Model")
        print("="*80)
        
        # 创建数据加载器（使用全部数据训练，不需要验证集）
        from cfp_dataset_loader import CFPLoader
        from torchvision import transforms
        
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
            split_ratio=1.0  # 使用全部数据
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        print(f"Training dataset size: {len(train_dataset)}")
        
        # 获取实际的类别数（从数据集）
        # Labels are mapped to 0-indexed values
        actual_num_classes = len(set(train_loader.dataset.labels))
        print(f"Actual number of classes in training data: {actual_num_classes}")
        
        # 重新创建模型，使用正确的类别数
        self.model = ArcFaceResNet50(
            num_classes=actual_num_classes,
            embedding_dim=self.embedding_dim,
            pretrained=True
        ).to(self.device)
        print(f"Model recreated with {actual_num_classes} classes")
        
        # 设置优化器和损失函数
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        
        # 训练循环
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # 获取logits（通过ArcFace层）
                features = self.model.backbone(images)
                features = self.model.global_pool(features)
                embeddings = self.model.bottleneck(features)
                logits = self.model.arcface(embeddings, labels)
                
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
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
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc': train_acc,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"  ✓ Model saved (Train Acc: {train_acc:.2f}%)")
            
            print("-" * 80)
        
        print(f"\n✓ Training completed! Final Train Acc: {train_acc:.2f}%")
        return train_acc
    
    def build_gallery(self, root_dir, save_path='../../galleries/gallery_features.pth'):
        """
        从 cfp-dataset 建立特征库
        
        Args:
            root_dir: cfp-dataset根目录
            save_path: 特征库保存路径
        """
        print("\n" + "="*80)
        print("Phase 2: Building Gallery Features")
        print("="*80)
        
        # 加载训练好的模型
        checkpoint = torch.load('../../models/best_model.pth')
        
        # 从checkpoint中获取类别数，重新创建模型
        num_classes = checkpoint['model_state_dict']['arcface.weight'].shape[0]
        print(f"Loading model with {num_classes} classes from checkpoint")
        
        self.model = ArcFaceResNet50(
            num_classes=num_classes,
            embedding_dim=self.embedding_dim,
            pretrained=False  # 不需要预训练，直接加载权重
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from checkpoint (epoch {checkpoint['epoch']})")
        
        # 创建数据集（使用所有图像，不split）
        from cfp_dataset_loader import CFPLoader
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        dataset = CFPLoader(
            root_dir=root_dir,
            mode='both',  # 使用正脸和侧脸
            transform=transform,
            train=True,
            split_ratio=1.0  # 使用全部数据
        )
        
        # 为每个人提取特征
        gallery_features = {}  # {person_id: embedding}
        
        # 获取所有唯一的人员ID（mapped labels）
        unique_labels = sorted(set(dataset.labels))
        print(f"Total persons in gallery: {len(unique_labels)}")
        
        # Create reverse mapping: mapped_label -> original_person_id
        reverse_mapping = {v: k for k, v in dataset.label_mapping.items()}
        print(f"Label mapping: {len(dataset.label_mapping)} persons")
        
        for mapped_label in tqdm(unique_labels, desc="Extracting features"):
            # Convert mapped label back to original person ID
            original_person_id = reverse_mapping[mapped_label]
            
            # 获取该人的所有图像索引
            indices = [i for i, label in enumerate(dataset.labels) if label == mapped_label]
            
            embeddings = []
            for idx in indices:
                image, _ = dataset[idx]
                image = image.unsqueeze(0).to(self.device)  # Add batch dimension
                
                with torch.no_grad():
                    # 提取embedding
                    emb = self.model.get_embedding(image)
                    # Remove batch dimension to get [512]
                    embeddings.append(emb.squeeze(0).cpu())
            
            # 取平均作为该人的特征
            if embeddings:
                avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
                # Store with original person_id (not mapped label)
                gallery_features[original_person_id] = avg_embedding
        
        # 保存特征库
        torch.save(gallery_features, save_path)
        print(f"\n✓ Gallery features saved to {save_path}")
        print(f"  Total persons: {len(gallery_features)}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        
        return gallery_features
    
    def test_recognition(self, occluded_root, gallery_path='../../galleries/gallery_features.pth', mode='both'):
        """
        在 occluded 数据集上测试识别准确率
        
        Args:
            occluded_root: occluded数据集根目录
            gallery_path: 特征库路径
            mode: 'frontal', 'profile', or 'both'
        
        Returns:
            accuracy: 识别准确率
        """
        print("\n" + "="*80)
        print("Phase 3: Testing Recognition on Occluded Dataset")
        print("="*80)
        
        # 加载模型
        checkpoint = torch.load('../../models/best_model.pth')
        
        # 从checkpoint中获取类别数，重新创建模型
        num_classes = checkpoint['model_state_dict']['arcface.weight'].shape[0]
        print(f"Loading model with {num_classes} classes from checkpoint")
        
        self.model = ArcFaceResNet50(
            num_classes=num_classes,
            embedding_dim=self.embedding_dim,
            pretrained=False
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 加载特征库
        gallery_features = torch.load(gallery_path)
        print(f"Gallery loaded: {len(gallery_features)} persons")
        
        # 加载occluded数据集
        occluded_loader = create_occluded_dataloader(
            root_dir=occluded_root,
            batch_size=32,
            mode=mode,
            num_workers=0
        )
        
        print(f"Testing on occluded dataset (mode={mode})")
        print(f"Total test images: {len(occluded_loader.dataset)}")
        
        # 测试
        correct = 0
        total = 0
        
        # 预计算所有gallery embeddings为一个tensor，加速计算
        gallery_ids = sorted(gallery_features.keys())
        print(f"Gallery contains {len(gallery_ids)} persons")
        
        # 确保所有 embedding 都是 tensor 并且维度正确
        gallery_embeddings_list = []
        for pid in gallery_ids:
            emb = gallery_features[pid]
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)
            # Ensure embedding is 1D [512], remove any extra dimensions
            emb = emb.squeeze()
            if emb.dim() == 0:
                # If scalar, reshape to [1]
                emb = emb.unsqueeze(0)
            gallery_embeddings_list.append(emb)
        
        gallery_tensor = torch.stack(gallery_embeddings_list).to(self.device)
        print(f"Gallery tensor shape: {gallery_tensor.shape}")
        
        progress_bar = tqdm(occluded_loader, desc="Testing")
        
        with torch.no_grad():
            for images, true_labels in progress_bar:
                images = images.to(self.device)
                
                # 批量提取embedding
                embeddings = self.model.get_embedding(images)  # [batch_size, 512]
                
                # 批量计算相似度
                # embeddings: [batch_size, 512]
                # gallery_tensor: [num_persons, 512]
                similarities = torch.nn.functional.cosine_similarity(
                    embeddings.unsqueeze(1),  # [batch_size, 1, 512]
                    gallery_tensor.unsqueeze(0),  # [1, num_persons, 512]
                    dim=2  # [batch_size, num_persons]
                )
                
                # 找到最相似的人
                _, predicted_indices = similarities.max(dim=1)
                predicted_ids = [gallery_ids[idx.item()] for idx in predicted_indices]
                
                # 统计正确数量
                for i, (pred_id, true_id) in enumerate(zip(predicted_ids, true_labels.tolist())):
                    # 只统计在gallery中存在的person
                    if true_id in gallery_features:
                        total += 1
                        if pred_id == true_id:
                            correct += 1
                    # 如果true_id不在gallery中，跳过（不应该发生）
                
                acc = 100. * correct / total
                progress_bar.set_postfix({'acc': f'{acc:.2f}%'})
        
        accuracy = 100. * correct / total
        
        print(f"\n{'='*80}")
        print(f"Recognition Results:")
        print(f"  Mode: {mode}")
        print(f"  Total samples: {total}")
        print(f"  Correct: {correct}")
        print(f"  Top-1 Accuracy: {accuracy:.2f}%")
        print(f"{'='*80}")
        
        return accuracy


def main():
    """主函数"""
    # 配置
    CFP_ROOT = r"d:\collegedata\2026\研二\警学项目申报\2d3dadvface\cfp-dataset"
    OCCLUDED_ROOT = r"d:\collegedata\2026\研二\警学项目申报\2d3dadvface\occluded"
    
    NUM_CLASSES = 500
    EMBEDDING_DIM = 512
    BATCH_SIZE = 32
    TRAIN_EPOCHS = 30
    LEARNING_RATE = 0.01  # 提高学习率到0.01
    
    # 创建系统
    system = FaceRecognitionSystem(
        num_classes=NUM_CLASSES,
        embedding_dim=EMBEDDING_DIM
    )
    
    # Phase 1: 训练模型
    system.train(
        root_dir=CFP_ROOT,
        epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        save_dir='../../models'
    )
    
    # Phase 2: 建立特征库
    system.build_gallery(
        root_dir=CFP_ROOT,
        save_path='../../galleries/gallery_features.pth'
    )
    
    # Phase 3: 测试识别（分别测试frontal和profile）
    acc_frontal = system.test_recognition(
        occluded_root=OCCLUDED_ROOT,
        gallery_path='../../galleries/gallery_features.pth',
        mode='frontal'
    )
    
    acc_profile = system.test_recognition(
        occluded_root=OCCLUDED_ROOT,
        gallery_path='../../galleries/gallery_features.pth',
        mode='profile'
    )
    
    acc_both = system.test_recognition(
        occluded_root=OCCLUDED_ROOT,
        gallery_path='../../galleries/gallery_features.pth',
        mode='both'
    )
    
    # 总结
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"Frontal recognition accuracy: {acc_frontal:.2f}%")
    print(f"Profile recognition accuracy: {acc_profile:.2f}%")
    print(f"Both views recognition accuracy: {acc_both:.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()

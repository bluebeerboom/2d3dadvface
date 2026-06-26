"""
AdaFace 人脸识别工具类
用于加载预训练模型、提取特征和进行人脸识别
"""
import torch
import numpy as np
import cv2
from PIL import Image
import os
import sys

# 添加 AdaFace-master 到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
adaface_path = os.path.join(project_root, 'AdaFace-master')
sys.path.insert(0, adaface_path)

from net import build_model


class AdaFaceRecognizer:
    """AdaFace 人脸识别器"""
    
    def __init__(self, model_path='models/adaface_ir50_ms1mv2.ckpt', device='cuda'):
        """
        初始化 AdaFace 识别器
        
        Args:
            model_path: 预训练模型路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[AdaFace] 使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("[AdaFace] 模型加载完成")
    
    def _load_model(self, model_path):
        """加载预训练模型"""
        model = build_model('ir_50')
        
        # 加载 checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # PyTorch 2.6+ 需要设置 weights_only=False
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 提取模型状态字典（去掉 'model.' 前缀）
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            model_state_dict = {
                key[6:]: val for key, val in state_dict.items() 
                if key.startswith('model.')
            }
        else:
            model_state_dict = checkpoint
        
        model.load_state_dict(model_state_dict)
        return model
    
    def preprocess_image(self, image):
        """
        预处理图像为 AdaFace 输入格式
        
        Args:
            image: numpy array (BGR format from OpenCV) or PIL Image
            
        Returns:
            torch.Tensor: 预处理后的张量 (1, 3, 112, 112)
        """
        # 如果是 OpenCV BGR 图像，转换为 RGB
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
            else:
                raise ValueError(f"不支持的图像形状: {image.shape}")
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("不支持的图像类型")
        
        # 调整大小为 112x112
        pil_image = pil_image.resize((112, 112), Image.BILINEAR)
        
        # 转换为 numpy 数组并归一化
        np_img = np.array(pil_image).astype(np.float32) / 255.0
        
        # 归一化: (x - 0.5) / 0.5
        normalized_img = (np_img - 0.5) / 0.5
        
        # 转换为 CHW 格式
        tensor = torch.from_numpy(normalized_img.transpose(2, 0, 1)).float()
        
        # 添加 batch 维度
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def extract_feature(self, image):
        """
        从图像中提取人脸特征
        
        Args:
            image: numpy array (BGR) or PIL Image
            
        Returns:
            numpy.ndarray: 512维特征向量
        """
        with torch.no_grad():
            # 预处理
            input_tensor = self.preprocess_image(image)
            input_tensor = input_tensor.to(self.device)
            
            # 提取特征
            feature, norm = self.model(input_tensor)
            
            # 转换为 numpy
            feature_np = feature.cpu().numpy().flatten()
            
            # L2 归一化
            feature_normalized = feature_np / (np.linalg.norm(feature_np) + 1e-8)
            
            return feature_normalized
    
    def extract_features_batch(self, images):
        """
        批量提取特征
        
        Args:
            images: list of numpy arrays or PIL Images
            
        Returns:
            numpy.ndarray: (N, 512) 特征矩阵
        """
        features = []
        for img in images:
            feat = self.extract_feature(img)
            features.append(feat)
        
        return np.array(features)
    
    def compute_similarity(self, feature1, feature2):
        """
        计算两个特征之间的余弦相似度
        
        Args:
            feature1: numpy.ndarray (512,)
            feature2: numpy.ndarray (512,)
            
        Returns:
            float: 余弦相似度 [-1, 1]
        """
        # 确保已归一化
        feat1_norm = feature1 / (np.linalg.norm(feature1) + 1e-8)
        feat2_norm = feature2 / (np.linalg.norm(feature2) + 1e-8)
        
        similarity = np.dot(feat1_norm, feat2_norm)
        return float(similarity)
    
    def recognize(self, query_feature, gallery_features, gallery_names, threshold=0.4):
        """
        识别人脸
        
        Args:
            query_feature: 查询特征 (512,)
            gallery_features: 图库特征矩阵 (N, 512)
            gallery_names: 图库名称列表 (N,)
            threshold: 识别阈值
            
        Returns:
            dict: {'name': 识别结果, 'similarity': 相似度, 'matched': 是否匹配}
        """
        similarities = []
        for gallery_feat in gallery_features:
            sim = self.compute_similarity(query_feature, gallery_feat)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # 找到最大相似度
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_name = gallery_names[best_idx]
        
        # 判断是否匹配
        matched = best_similarity >= threshold
        
        return {
            'name': best_name if matched else 'Unknown',
            'similarity': best_similarity,
            'matched': matched,
            'all_similarities': similarities
        }


if __name__ == '__main__':
    # 测试代码
    print("="*70)
    print("AdaFace Recognizer 测试")
    print("="*70)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, 'models', 'adaface_ir50_ms1mv2.ckpt')
    
    if os.path.exists(model_path):
        try:
            recognizer = AdaFaceRecognizer(model_path=model_path)
            print("\n✓ AdaFace Recognizer 初始化成功")
        except Exception as e:
            print(f"\n❌ 初始化失败: {e}")
    else:
        print(f"\n❌ 模型文件不存在: {model_path}")

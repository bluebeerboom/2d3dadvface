"""
Face Detection Module
人脸检测模块

支持多种人脸检测算法：
- MediaPipe Face (推荐，轻量高效)
- OpenCV Haar Cascade (备选，无需额外依赖)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """
    人脸检测器基类
    """
    
    def detect_faces(self, image_rgb: np.ndarray) -> List[dict]:
        """
        检测人脸
        
        Args:
            image_rgb: RGB格式的图像 (H, W, 3)
            
        Returns:
            List of face detections, each containing:
            - box: [x1, y1, x2, y2] 人脸边界框
            - confidence: 置信度
            - landmarks: 关键点（可选）
        """
        raise NotImplementedError


class ImprovedHaarDetector(FaceDetector):
    """
    改进的 Haar Cascade 人脸检测器
    
    优点：
    - 无需额外依赖
    - 通过多尺度检测和参数优化提高小图检测率
    - 对正脸效果良好
    """
    
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 3, min_size: tuple = (30, 30)):
        """
        Args:
            scale_factor: 图像缩放比例 (越小越精确但越慢)
            min_neighbors: 最小邻居数 (越小召回率越高但误检越多)
            min_size: 最小人脸尺寸
        """
        try:
            # Load Haar Cascade classifier
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            self.scale_factor = scale_factor
            self.min_neighbors = min_neighbors
            self.min_size = min_size
            
            print("✓ Improved Haar Cascade detector initialized")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Haar Cascade: {e}")
    
    def detect_faces(self, image_rgb: np.ndarray) -> List[dict]:
        """
        使用改进的 Haar Cascade 检测人脸
        
        Args:
            image_rgb: RGB图像 (H, W, 3)
            
        Returns:
            检测到的人脸列表
        """
        try:
            h, w = image_rgb.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Equalize histogram for better contrast
            gray_eq = cv2.equalizeHist(gray)
            
            # Detect faces with optimized parameters
            faces_rect = self.face_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            detections = []
            
            for (x, y, box_w, box_h) in faces_rect:
                x2 = x + box_w
                y2 = y + box_h
                
                # Ensure within bounds
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Estimate confidence based on face size and position
                # Larger faces in center get higher confidence
                area_ratio = (box_w * box_h) / (w * h)
                center_x = abs((x + x2) / 2 - w / 2) / w
                center_y = abs((y + y2) / 2 - h / 2) / h
                
                # Simple heuristic for confidence
                confidence = min(1.0, area_ratio * 10) * (1.0 - (center_x + center_y) / 2)
                
                detections.append({
                    'box': [x, y, x2, y2],
                    'confidence': float(confidence),
                    'landmarks': None
                })
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detections
            
        except Exception as e:
            print(f"Error in Improved Haar detection: {e}")
            return []


class HaarCascadeDetector(FaceDetector):
    """
    OpenCV Haar Cascade 人脸检测器
    
    优点：
    - 速度快
    - 无额外依赖
    
    缺点：
    - 对遮挡人脸效果差
    - 只能检测正面人脸
    """
    
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5):
        """
        Args:
            scale_factor: 图像缩放比例
            min_neighbors: 最小邻居数
        """
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        print("✓ Haar Cascade detector initialized")
    
    def detect_faces(self, image_rgb: np.ndarray) -> List[dict]:
        """
        使用 Haar Cascade 检测人脸
        
        Args:
            image_rgb: RGB图像 (H, W, 3)
            
        Returns:
            检测到的人脸列表
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(50, 50),
                maxSize=(image_rgb.shape[1], image_rgb.shape[0])
            )
            
            if len(faces) == 0:
                return []
            
            # Convert to standard format
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    'box': [x, y, x + w, y + h],  # [x1, y1, x2, y2]
                    'confidence': 1.0,  # Haar Cascade doesn't provide confidence
                    'landmarks': None
                })
            
            # Sort by area (largest first)
            detections.sort(
                key=lambda x: (x['box'][2] - x['box'][0]) * (x['box'][3] - x['box'][1]),
                reverse=True
            )
            
            return detections
            
        except Exception as e:
            print(f"Error in Haar Cascade detection: {e}")
            return []


def create_face_detector(detector_type: str = 'improved_haar', **kwargs) -> FaceDetector:
    """
    工厂函数：创建人脸检测器
    
    Args:
        detector_type: 检测器类型 ('improved_haar', 'haar')
        **kwargs: 传递给检测器的参数
        
    Returns:
        FaceDetector 实例
    """
    if detector_type.lower() == 'improved_haar':
        return ImprovedHaarDetector(**kwargs)
    elif detector_type.lower() == 'haar':
        return HaarCascadeDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


if __name__ == "__main__":
    # 测试代码
    import os
    
    print("=" * 80)
    print("Testing Face Detectors")
    print("=" * 80)
    
    # 加载测试图像
    test_image_path = r"d:\collegedata\2026\研二\警学项目申报\2d3dadvface\occluded\001\frontal\001_01.jpg"
    
    if os.path.exists(test_image_path):
        img_array = np.fromfile(test_image_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"\nTest image size: {image_rgb.shape}")
        
        # 测试 Improved Haar
        print("\n--- Testing Improved Haar ---")
        try:
            detector = create_face_detector('improved_haar', scale_factor=1.1, min_neighbors=3, min_size=(30, 30))
            faces = detector.detect_faces(image_rgb)
            print(f"Detected {len(faces)} face(s)")
            for i, face in enumerate(faces):
                print(f"  Face {i+1}: box={face['box']}, confidence={face['confidence']:.2f}")
        except Exception as e:
            print(f"Improved Haar test failed: {e}")
        
        # 测试 Haar Cascade
        print("\n--- Testing Haar Cascade ---")
        try:
            detector = create_face_detector('haar')
            faces = detector.detect_faces(image_rgb)
            print(f"Detected {len(faces)} face(s)")
            for i, face in enumerate(faces):
                print(f"  Face {i+1}: box={face['box']}, confidence={face['confidence']:.2f}")
        except Exception as e:
            print(f"Haar Cascade test failed: {e}")
    
    print("\n✓ Face detector tests completed!")

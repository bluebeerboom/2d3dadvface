"""
Utility Modules
工具模块包

包含：
- face_detector: 人脸检测器 (Improved Haar, Standard Haar)
- face_processing: 人脸处理工具
- adaface_recognizer: AdaFace 人脸识别器
"""

from .face_detector import (
    FaceDetector,
    ImprovedHaarDetector,
    HaarCascadeDetector,
    create_face_detector
)

from .face_processing import (
    crop_face_with_padding,
    draw_face_detection_result,
    select_best_face
)

from .adaface_recognizer import AdaFaceRecognizer

__all__ = [
    'FaceDetector',
    'ImprovedHaarDetector', 
    'HaarCascadeDetector',
    'create_face_detector',
    'crop_face_with_padding',
    'draw_face_detection_result',
    'select_best_face',
    'AdaFaceRecognizer'
]

"""
Face Recognition System Package
门禁人脸识别系统

Package structure:
- models/: ArcFace model implementation
- datasets/: Dataset loaders (CFP, Occluded)
- scripts/: Training and testing scripts
- utils/: Utility functions
"""

# 不在包初始化时强制导入，避免子模块缺失导致整个包不可用
# 需要时直接: from src.models.models.arcface_model import ArcFaceResNet50
# 需要时直接: from src.datasets import CFPLoader, OccludedLoader

__version__ = '2.0.0'
__author__ = 'Your Name'

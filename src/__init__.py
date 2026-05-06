"""
Face Recognition System Package
门禁人脸识别系统

Package structure:
- models/: ArcFace model implementation
- datasets/: Dataset loaders (CFP, Occluded)
- scripts/: Training and testing scripts
- utils/: Utility functions
"""

from .models import ArcFaceResNet50
from .datasets import CFPLoader, OccludedLoader, create_cfp_dataloaders, create_occluded_dataloader

__version__ = '2.0.0'
__author__ = 'Your Name'

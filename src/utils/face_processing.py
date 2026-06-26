"""
Face Processing Utilities
人脸处理工具模块

提供人脸裁剪、预处理等功能
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def crop_face_with_padding(
    image_rgb: np.ndarray,
    box: list,
    padding_ratio: float = 0.2
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    根据边界框裁剪人脸区域，并添加padding
    
    Args:
        image_rgb: RGB图像 (H, W, 3)
        box: 人脸边界框 [x1, y1, x2, y2]
        padding_ratio: padding比例（相对于人脸尺寸）
        
    Returns:
        (cropped_face, original_box)
        - cropped_face: 裁剪后的人脸图像
        - original_box: 原始边界框 (x, y, w, h)
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = box
    
    # Calculate face dimensions
    face_w = x2 - x1
    face_h = y2 - y1
    
    # Add padding
    padding = int(max(face_w, face_h) * padding_ratio)
    x1_padded = max(0, x1 - padding)
    y1_padded = max(0, y1 - padding)
    x2_padded = min(w, x2 + padding)
    y2_padded = min(h, y2 + padding)
    
    # Crop face region
    face_img = image_rgb[y1_padded:y2_padded, x1_padded:x2_padded]
    
    return face_img, (x1, y1, face_w, face_h)


def draw_face_detection_result(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制人脸检测结果
    
    Args:
        image: BGR图像
        box: 边界框 (x, y, w, h)
        label: 标签文本
        color: 颜色 (B, G, R)
        thickness: 线条粗细
        
    Returns:
        绘制后的图像
    """
    result_img = image.copy()
    x, y, w, h = box
    
    # Draw rectangle
    cv2.rectangle(result_img, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label
    if label:
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            result_img,
            (x, y - text_h - baseline - 5),
            (x + text_w, y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            result_img,
            label,
            (x, y - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),  # White text
            text_thickness
        )
    
    return result_img


def select_best_face(
    detections: list,
    criteria: str = 'confidence'
) -> Optional[dict]:
    """
    从多个人脸检测中选择最佳的一个
    
    Args:
        detections: 人脸检测列表
        criteria: 选择标准 ('confidence', 'largest', 'center')
        
    Returns:
        最佳人脸检测，如果没有检测到则返回 None
    """
    if not detections:
        return None
    
    if criteria == 'confidence':
        # 选择置信度最高的
        return max(detections, key=lambda x: x.get('confidence', 0))
    
    elif criteria == 'largest':
        # 选择面积最大的
        return max(
            detections,
            key=lambda x: (x['box'][2] - x['box'][0]) * (x['box'][3] - x['box'][1])
        )
    
    elif criteria == 'center':
        # 选择最靠近图像中心的
        # This would require image dimensions, so we'll skip for now
        return detections[0]
    
    else:
        return detections[0]


if __name__ == "__main__":
    # 测试代码
    print("=" * 80)
    print("Testing Face Processing Utilities")
    print("=" * 80)
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test crop_face_with_padding
    print("\n--- Testing crop_face_with_padding ---")
    test_box = [100, 100, 200, 200]  # [x1, y1, x2, y2]
    cropped, original_box = crop_face_with_padding(dummy_image, test_box, padding_ratio=0.2)
    print(f"Original box: {test_box}")
    print(f"Cropped face shape: {cropped.shape}")
    print(f"Original box returned: {original_box}")
    
    # Test draw_face_detection_result
    print("\n--- Testing draw_face_detection_result ---")
    result_img = draw_face_detection_result(
        dummy_image,
        (100, 100, 100, 100),
        label="Test Person (0.95)",
        color=(0, 255, 0)
    )
    print(f"Result image shape: {result_img.shape}")
    
    # Test select_best_face
    print("\n--- Testing select_best_face ---")
    test_detections = [
        {'box': [10, 10, 50, 50], 'confidence': 0.7},
        {'box': [100, 100, 200, 200], 'confidence': 0.9},
        {'box': [300, 300, 350, 350], 'confidence': 0.8},
    ]
    
    best_confidence = select_best_face(test_detections, 'confidence')
    print(f"Best by confidence: {best_confidence}")
    
    best_largest = select_best_face(test_detections, 'largest')
    print(f"Best by largest: {best_largest}")
    
    print("\n✓ Face processing utilities tests completed!")

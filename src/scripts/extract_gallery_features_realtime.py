"""
从 Dataset 数据库提取 AdaFace 特征并构建特征库
使用 InsightFace 进行人脸检测和对齐，提高特征质量
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 先导入 cv2，再导入 insightface（避免版本冲突）
import cv2
import insightface
from insightface.app import FaceAnalysis

from src.utils.adaface_recognizer import AdaFaceRecognizer


def extract_dataset_features(database_path, output_path, 
                             model_path='models/adaface_ir50_ms1mv2.ckpt'):
    """
    从 Dataset 数据库提取人脸特征
    
    Args:
        database_path: 数据库路径，结构如下：
            database_path/
            ├── 001/
            │   ├── photo1.jpg
            │   ├── photo2.jpg
            │   └── ...
            ├── 002/
            │   └── ...
            └── ...
        output_path: 输出特征库路径
        model_path: AdaFace 模型路径
    """
    print("="*70)
    print("Dataset 数据库特征提取（InsightFace 检测 + AdaFace 识别）")
    print("="*70)
    
    # 初始化 InsightFace 检测器
    print("\n[1] 加载 InsightFace 检测器...")
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("  ✓ 使用 GPU 加速")
    else:
        providers = ['CPUExecutionProvider']
        print("  ⚠ 使用 CPU")
    
    detector = FaceAnalysis(providers=providers)
    detector.prepare(ctx_id=0, det_size=(640, 640))
    print("  ✓ InsightFace 检测器加载完成")
    
    # 初始化 AdaFace 识别器
    print("\n[2] 加载 AdaFace 模型...")
    recognizer = AdaFaceRecognizer(model_path=model_path)
    print("  ✓ AdaFace 模型加载完成")
    
    # 获取所有人员文件夹
    database_dir = Path(database_path)
    person_folders = sorted([d for d in database_dir.iterdir() if d.is_dir()])
    
    print(f"\n找到 {len(person_folders)} 个人")
    
    # 存储特征和标签
    all_features = []
    all_names = []
    all_person_ids = []
    
    # 遍历每个人
    for person_folder in tqdm(person_folders, desc="提取特征"):
        person_id = person_folder.name  # 文件夹名作为ID (001, 002, ...)
        person_name = f"Person_{person_id}"  # 显示名称
        
        # 收集该人的所有图像
        person_images = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        
        for ext in image_extensions:
            for img_path in person_folder.glob(ext):
                try:
                    # 使用 numpy + cv2.imdecode 解决中文路径问题
                    img_array = np.fromfile(str(img_path), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        person_images.append((img, img_path.name))
                except Exception as e:
                    print(f"  ⚠ 读取失败 {img_path}: {e}")
        
        if len(person_images) == 0:
            print(f"  ⚠ {person_name}: 没有有效图像")
            continue
        
        print(f"\n处理: {person_name} ({len(person_images)} 张照片)")
        
        # 提取该人所有图像的特征
        person_features = []
        success_count = 0
        fail_count = 0
        
        for img, img_name in person_images:
            try:
                # 步骤1: 使用 InsightFace 检测和对齐人脸
                faces = detector.get(img)
                
                if len(faces) == 0:
                    print(f"  ⚠ {img_name}: 未检测到人脸")
                    fail_count += 1
                    continue
                
                # 选择置信度最高的人脸
                best_face = max(faces, key=lambda f: f.det_score)
                
                # 获取对齐后的人脸图像
                aligned_face = best_face.normed_embedding  # 这已经是归一化的嵌入
                
                # 如果需要裁剪对齐的人脸图像，可以使用 bbox
                bbox = best_face.bbox.astype(np.int32)
                x1, y1, x2, y2 = bbox
                
                # 添加边距并裁剪
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(img.shape[1], x2 + margin)
                y2 = min(img.shape[0], y2 + margin)
                
                face_img = img[y1:y2, x1:x2]
                
                if face_img.size == 0:
                    print(f"  ⚠ {img_name}: 裁剪后图像为空")
                    fail_count += 1
                    continue
                
                # 步骤2: 使用 AdaFace 提取高质量特征
                feat = recognizer.extract_feature(face_img)
                person_features.append(feat)
                success_count += 1
                print(f"  ✓ {img_name} (det_score: {best_face.det_score:.2f})")
                
            except Exception as e:
                print(f"  ⚠ {img_name}: 特征提取失败 - {e}")
                fail_count += 1
        
        print(f"  成功: {success_count}, 失败: {fail_count}")
        
        if len(person_features) > 0:
            # 计算平均特征（更鲁棒）
            avg_feature = np.mean(person_features, axis=0)
            # L2 归一化
            avg_feature = avg_feature / (np.linalg.norm(avg_feature) + 1e-8)
            
            all_features.append(avg_feature)
            all_names.append(person_name)
            all_person_ids.append(person_id)
            
            print(f"  ✓ {person_name} 特征提取完成")
    
    # 转换为 numpy 数组
    if len(all_features) == 0:
        print("\n❌ 没有成功提取任何特征")
        return None, None, None
    
    all_features = np.array(all_features)
    
    print(f"\n{'='*70}")
    print(f"✓ 成功提取 {len(all_features)} 个人的特征")
    print(f"  特征维度: {all_features.shape}")
    print(f"  人员名单: {', '.join(all_names)}")
    
    # 保存特征库
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存特征和元数据
    np.savez_compressed(
        output_path,
        features=all_features,
        names=np.array(all_names),
        person_ids=np.array(all_person_ids)
    )
    
    # 保存元数据为 JSON
    metadata = {
        'num_persons': len(all_person_ids),
        'feature_dim': int(all_features.shape[1]),
        'person_ids': all_person_ids,
        'names': all_names,
        'database_path': str(database_path)
    }
    
    metadata_path = output_path.replace('.npz', '_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 特征库已保存:")
    print(f"  特征文件: {output_path}")
    print(f"  元数据文件: {metadata_path}")
    
    return all_features, all_names, all_person_ids


if __name__ == '__main__':
    # 配置路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Dataset 数据库路径
    database_path = os.path.join(project_root, 'Dataset')
    
    # 输出路径 - galleries 文件夹
    output_path = os.path.join(project_root, 'galleries', 'gallery_features_realtime.npz')
    
    # 模型路径
    model_path = os.path.join(project_root, 'models', 'adaface_ir50_ms1mv2.ckpt')
    
    # 检查数据库是否存在
    if not os.path.exists(database_path):
        print(f"❌ 数据库不存在: {database_path}")
        sys.exit(1)
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        sys.exit(1)
    
    # 提取特征
    features, names, person_ids = extract_dataset_features(
        database_path=database_path,
        output_path=output_path,
        model_path=model_path
    )
    
    if features is not None:
        print("\n" + "="*70)
        print("特征提取完成！")
        print("="*70)
        print(f"\n现在可以运行实时识别系统了")

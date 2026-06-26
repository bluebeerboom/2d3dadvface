"""
从 Dataset 数据库提取高质量人脸特征并构建特征库 (v2)
=====================================================
核心改进：
  1. 5点关键点对齐 → 相似变换到 ArcFace 标准模板（解决相似度 0.35 的根本原因）
  2. InsightFace ArcFace 嵌入 (512d) + 对齐后 AdaFace 特征 (512d) → 双通道
  3. 质量筛选：det_score < 0.7 自动过滤
  4. 鲁棒聚合：中位数 + 质量加权均值，抗离群值
  5. 对齐一致性：提取和推理使用完全相同的对齐管线
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import cv2
import insightface
from insightface.app import FaceAnalysis
from src.utils.adaface_recognizer import AdaFaceRecognizer


# ===== ArcFace 112x112 标准 5 点模板 =====
# 顺序: [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]
ARCFACE_5POINT_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655]
], dtype=np.float32)

# 对齐后的人脸尺寸
ALIGN_SIZE = 112


def align_face_with_kps(img_bgr, kps, target_size=ALIGN_SIZE):
    """
    使用 5 点关键点做相似变换对齐人脸到标准模板

    Args:
        img_bgr: BGR 图像 (H, W, 3)
        kps: 5 个关键点 (5, 2)，顺序 [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]
        target_size: 输出尺寸

    Returns:
        aligned_face: 对齐后的人脸 (target_size, target_size, 3)
        success: 是否成功
    """
    if kps is None or kps.shape[0] < 5:
        return None, False

    kps = kps.astype(np.float32)

    # 计算从 kps 到标准模板的相似变换矩阵 (仅旋转+缩放+平移, 无剪切)
    # RANSAC 抗关键点噪声
    M, inliers = cv2.estimateAffinePartial2D(
        kps, ARCFACE_5POINT_TEMPLATE,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )

    if M is None:
        return None, False

    # 应用仿射变换
    aligned = cv2.warpAffine(img_bgr, M, (target_size, target_size),
                             borderMode=cv2.BORDER_REPLICATE)

    return aligned, True


def quality_filter(det_score, kps=None, face_area_ratio=0.0):
    """
    人脸质量过滤

    返回 True 表示通过质量检查
    """
    # 检测置信度过低
    if det_score < 0.7:
        return False, "det_score too low"

    # 关键点完整性
    if kps is not None:
        if kps.shape[0] < 5:
            return False, "insufficient keypoints"
        # 检查关键点是否在合理范围
        if np.any(kps < 0):
            return False, "kps out of bounds"

    return True, "ok"


def extract_dataset_features_v2(database_path, output_dir,
                                 model_path='models/adaface_ir50_ms1mv2.ckpt'):
    """
    从 Dataset 提取高质量人脸特征 (v2 对齐版)

    输出:
      - gallery_arcface.npz     : InsightFace 内置 ArcFace 特征 (主通道)
      - gallery_adaface.npz     : 对齐后 AdaFace 特征 (辅助通道)
      - gallery_meta.json       : 元数据
    """
    print("=" * 70)
    print("Dataset 特征提取 v2 (5点对齐 + 双通道融合)")
    print("=" * 70)

    # ---- [1] 加载 InsightFace ----
    print("\n[1] 加载 InsightFace 检测器...")
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("  ✓ GPU 加速 (CUDA)")
    else:
        providers = ['CPUExecutionProvider']
        print("  ⚠ CPU 模式")

    detector = FaceAnalysis(
        providers=providers,
        allowed_modules=['detection', 'recognition']  # 只需要检测+识别
    )
    detector.prepare(ctx_id=0, det_size=(640, 640))
    print("  ✓ InsightFace 加载完成 (含 ArcFace w600k_r50)")

    # ---- [2] 加载 AdaFace ----
    print("\n[2] 加载 AdaFace 模型...")
    recognizer = AdaFaceRecognizer(model_path=model_path)
    print("  ✓ AdaFace IR-50 加载完成")

    # ---- [3] 扫描人员 ----
    database_dir = Path(database_path)
    person_folders = sorted([d for d in database_dir.iterdir() if d.is_dir()])
    print(f"\n[3] 扫描到 {len(person_folders)} 个人员文件夹")

    # 存储结果
    arcface_features = []   # InsightFace ArcFace (512d)
    adaface_features = []   # 对齐后 AdaFace (512d)
    names = []
    person_ids = []
    quality_stats = []

    # ---- [4] 遍历提取 ----
    for person_folder in person_folders:
        pid = person_folder.name
        # 中文名直接用文件夹名，数字ID加前缀
        pname = pid if any('\u4e00' <= c <= '\u9fff' for c in pid) else f"Person_{pid}"

        # 收集图像
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            for img_path in person_folder.glob(ext):
                try:
                    arr = np.fromfile(str(img_path), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        images.append((img, img_path.name))
                except Exception:
                    pass

        if not images:
            print(f"\n  ⚠ {pname}: 无有效图像，跳过")
            continue

        print(f"\n{'─' * 50}")
        print(f"  处理: {pname} ({len(images)} 张)")

        person_arc = []   # ArcFace 特征
        person_ada = []   # AdaFace 特征
        person_weights = []  # 质量权重

        for img, img_name in images:
            try:
                faces = detector.get(img)
                if not faces:
                    print(f"    ✗ {img_name}: 未检测到人脸")
                    continue

                # 取最高置信度的人脸
                best = max(faces, key=lambda f: f.det_score)

                # 质量过滤
                qualified, reason = quality_filter(best.det_score, best.kps)
                if not qualified:
                    print(f"    ✗ {img_name}: {reason} (det={best.det_score:.2f})")
                    continue

                # ===== 通道1: InsightFace ArcFace 嵌入 (已对齐) =====
                arc_emb = best.normed_embedding  # 512-dim, L2 normalized
                if arc_emb is None:
                    print(f"    ✗ {img_name}: ArcFace 嵌入为空")
                    continue

                # ===== 通道2: 5点对齐 → AdaFace =====
                aligned_face, ok = align_face_with_kps(img, best.kps)
                if not ok:
                    print(f"    ✗ {img_name}: 对齐失败")
                    continue

                ada_emb = recognizer.extract_feature(aligned_face)

                # 质量权重 = det_score
                weight = float(best.det_score)

                person_arc.append(arc_emb)
                person_ada.append(ada_emb)
                person_weights.append(weight)

                print(f"    ✓ {img_name}  arc={np.linalg.norm(arc_emb):.3f}  "
                      f"ada={np.linalg.norm(ada_emb):.3f}  det={best.det_score:.2f}")

            except Exception as e:
                print(f"    ✗ {img_name}: {e}")

        if not person_arc:
            print(f"  ⚠ {pname}: 无有效特征，跳过")
            continue

        # ---- 聚合: 质量加权中位数 ----
        # 使用 median 比 mean 更抗离群值，再加质量权重

        # 按权重排序，取中位数附近的高质量特征
        arc_array = np.array(person_arc)
        ada_array = np.array(person_ada)
        weights = np.array(person_weights)

        # 方法: 质量加权均值 vs 中位数，选更鲁棒的
        # 如果样本数 >= 5，用质量加权均值；否则用中位数
        if len(person_arc) >= 5:
            # 质量加权均值
            w_norm = weights / (weights.sum() + 1e-8)
            arc_final = (arc_array * w_norm[:, np.newaxis]).sum(axis=0)
            ada_final = (ada_array * w_norm[:, np.newaxis]).sum(axis=0)
            agg_method = "weighted-mean"
        else:
            # 中位数
            arc_final = np.median(arc_array, axis=0)
            ada_final = np.median(ada_array, axis=0)
            agg_method = "median"

        # L2 归一化
        arc_final /= (np.linalg.norm(arc_final) + 1e-8)
        ada_final /= (np.linalg.norm(ada_final) + 1e-8)

        arcface_features.append(arc_final)
        adaface_features.append(ada_final)
        names.append(pname)
        person_ids.append(pid)

        # 计算类内一致性 (自相似度)
        if len(person_arc) >= 2:
            intra_sims = []
            for i in range(len(person_arc)):
                for j in range(i + 1, len(person_arc)):
                    sim = np.dot(person_arc[i], person_arc[j])
                    intra_sims.append(sim)
            avg_intra = np.mean(intra_sims)
        else:
            avg_intra = 1.0

        quality_stats.append({
            'name': pname, 'id': pid,
            'num_images': len(person_arc),
            'agg_method': agg_method,
            'avg_intra_sim': float(avg_intra),
            'avg_det_score': float(np.mean(weights))
        })

        print(f"  ✓ {pname}: {len(person_arc)} 张 → {agg_method} "
              f"(类内相似度={avg_intra:.3f}, 平均det={np.mean(weights):.2f})")

    # ---- [5] 保存 ----
    n_persons = len(arcface_features)
    if n_persons == 0:
        print("\n❌ 无有效特征，退出")
        return

    arcface_features = np.array(arcface_features)
    adaface_features = np.array(adaface_features)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 ArcFace 特征库
    np.savez_compressed(
        str(output_dir / 'gallery_arcface.npz'),
        features=arcface_features,
        names=np.array(names),
        person_ids=np.array(person_ids)
    )

    # 保存 AdaFace 特征库
    np.savez_compressed(
        str(output_dir / 'gallery_adaface.npz'),
        features=adaface_features,
        names=np.array(names),
        person_ids=np.array(person_ids)
    )

    # 元数据
    metadata = {
        'version': 'v2',
        'num_persons': n_persons,
        'feature_dim': int(arcface_features.shape[1]),
        'person_ids': person_ids,
        'names': names,
        'database_path': str(database_path),
        'quality_stats': quality_stats,
        'alignment': {
            'method': '5-point similarity transform (cv2.estimateAffinePartial2D)',
            'template': 'ArcFace 112x112 canonical',
            'align_size': ALIGN_SIZE
        },
        'features': {
            'arcface': 'InsightFace buffalo_l w600k_r50 (512d)',
            'adaface': 'AdaFace IR-50 with 5-point alignment (512d)'
        }
    }

    with open(str(output_dir / 'gallery_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ---- [6] 质量报告 ----
    print(f"\n{'=' * 70}")
    print(f"✓ 特征提取完成! 共 {n_persons} 人")
    print(f"{'=' * 70}")
    print(f"\n{'人员':<12} {'图片数':<8} {'类内相似度':<12} {'平均det':<10} {'聚合方式'}")
    print(f"{'─' * 55}")
    for s in quality_stats:
        flag = "⚠" if s['avg_intra_sim'] < 0.5 else "✓"
        print(f"{flag} {s['name']:<10} {s['num_images']:<8} "
              f"{s['avg_intra_sim']:<12.3f} {s['avg_det_score']:<10.2f} {s['agg_method']}")

    # 类间相似度检查 (检查是否有混淆风险)
    print(f"\n{'=' * 70}")
    print("类间相似度检查 (ArcFace 通道)")
    print(f"{'=' * 70}")
    confusion_pairs = []
    for i in range(n_persons):
        for j in range(i + 1, n_persons):
            sim = np.dot(arcface_features[i], arcface_features[j])
            if sim > 0.35:  # 类间相似度过高 → 可能混淆
                confusion_pairs.append((names[i], names[j], sim))

    if confusion_pairs:
        confusion_pairs.sort(key=lambda x: -x[2])
        print(f"⚠ 发现 {len(confusion_pairs)} 对高相似度组合 (>0.35):")
        for n1, n2, sim in confusion_pairs:
            print(f"  {n1} ↔ {n2}: {sim:.3f}")
    else:
        print("✓ 所有类间相似度 < 0.35，区分度良好")

    print(f"\n输出文件:")
    print(f"  ArcFace  : {output_dir / 'gallery_arcface.npz'}")
    print(f"  AdaFace  : {output_dir / 'gallery_adaface.npz'}")
    print(f"  元数据   : {output_dir / 'gallery_meta.json'}")

    return arcface_features, names, person_ids


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    database_path = os.path.join(project_root, 'Dataset')
    output_dir = os.path.join(project_root, 'galleries')
    model_path = os.path.join(project_root, 'models', 'adaface_ir50_ms1mv2.ckpt')

    if not os.path.exists(database_path):
        print(f"❌ 数据库不存在: {database_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        sys.exit(1)

    extract_dataset_features_v2(
        database_path=database_path,
        output_dir=output_dir,
        model_path=model_path
    )

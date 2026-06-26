"""
基于 Intel RealSense D455 的实时人脸识别系统 (v2)
================================================
改进：
  1. 5点关键点对齐 → 与特征库提取完全一致的对齐管线
  2. 双通道融合：ArcFace (主) + AdaFace (辅)
  3. 自适应阈值：根据 top1/top2 差距动态判断
"""
import os
import sys
import json
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrealsense2 as rs
import insightface
from insightface.app import FaceAnalysis

from src.utils.adaface_recognizer import AdaFaceRecognizer

# ===== 对齐模板 (与特征库提取完全一致) =====
ARCFACE_5POINT_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655]
], dtype=np.float32)
ALIGN_SIZE = 112


# ─── 中文字体 ─────────────────────────────────────────────────
def _get_chinese_font(size):
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def put_chinese_text(img, text, pos, font_size, color_bgr):
    if not text:
        return
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _get_chinese_font(font_size)
    draw.text(pos, text, font=font, fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    np.copyto(img, result)


def put_chinese_text_bg(img, text, pos, font_size, color_bgr, bg_bgr):
    if not text:
        return
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _get_chinese_font(font_size)
    cr = (color_bgr[2], color_bgr[1], color_bgr[0])
    br = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.rectangle([pos[0] - 2, pos[1], pos[0] + tw + 4, pos[1] + th + 4], fill=br)
    draw.text((pos[0] + 2, pos[1] + 1), text, font=font, fill=cr)
    result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    np.copyto(img, result)


def clean_name(name):
    """去掉 'Person_' 前缀"""
    s = str(name)
    return s[7:] if s.startswith("Person_") else s


def align_face_with_kps(img_bgr, kps):
    """5点对齐 (与 extract 脚本完全一致)"""
    if kps is None or kps.shape[0] < 5:
        return None
    kps = kps.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(
        kps, ARCFACE_5POINT_TEMPLATE,
        method=cv2.RANSAC, ransacReprojThreshold=5.0
    )
    if M is None:
        return None
    return cv2.warpAffine(img_bgr, M, (ALIGN_SIZE, ALIGN_SIZE),
                          borderMode=cv2.BORDER_REPLICATE)


class RealTimeFaceRecognitionV2:
    """实时人脸识别系统 v2 - 对齐 + 双通道融合"""

    def __init__(self, gallery_dir='galleries',
                 model_path='models/adaface_ir50_ms1mv2.ckpt',
                 threshold=0.45,
                 arcface_weight=0.65):
        """
        Args:
            gallery_dir: 特征库目录 (含 gallery_arcface.npz, gallery_adaface.npz)
            model_path: AdaFace 模型路径
            threshold: 基础识别阈值
            arcface_weight: ArcFace 通道融合权重 (0~1)
        """
        print("=" * 70)
        print("实时人脸识别系统 v2 (5点对齐 + 双通道融合)")
        print("=" * 70)

        self.threshold = threshold
        self.arcface_weight = arcface_weight
        self.adaface_weight = 1.0 - arcface_weight

        # ---- [1] 加载特征库 ----
        print("\n[1] 加载特征库...")
        arcface_path = os.path.join(gallery_dir, 'gallery_arcface.npz')
        adaface_path = os.path.join(gallery_dir, 'gallery_adaface.npz')

        self.use_dual_channel = (
            os.path.exists(arcface_path) and os.path.exists(adaface_path)
        )

        if self.use_dual_channel:
            arc_data = np.load(arcface_path, allow_pickle=True)
            ada_data = np.load(adaface_path, allow_pickle=True)
            self.gallery_arc = arc_data['features']
            self.gallery_ada = ada_data['features']
            self.gallery_names = arc_data['names']
            self.gallery_ids = arc_data['person_ids']
            print(f"  ✓ 双通道模式: ArcFace + AdaFace")
        else:
            # 回退到单通道
            fallback = os.path.join(gallery_dir, 'gallery_features_realtime.npz')
            data = np.load(fallback, allow_pickle=True)
            self.gallery_arc = data['features']
            self.gallery_ada = None
            self.gallery_names = data['names']
            self.gallery_ids = data['person_ids']
            self.use_dual_channel = False
            print(f"  ⚠ 单通道模式 (回退到旧特征库)")

        print(f"  ✓ {len(self.gallery_names)} 人已加载")

        # ---- [2] AdaFace ----
        print("\n[2] 加载 AdaFace 模型...")
        self.recognizer = AdaFaceRecognizer(model_path=model_path, device='cuda')

        # ---- [3] InsightFace ----
        print("\n[3] 加载 InsightFace 检测器...")
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if \
            'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']

        self.detector = FaceAnalysis(
            providers=providers,
            allowed_modules=['detection', 'recognition']
        )
        self.detector.prepare(ctx_id=0, det_size=(640, 640))

        # ---- 统计 ----
        self.frame_count = 0
        self.fps_start = time.time()
        self.current_fps = 0

        print("\n[4] 系统就绪")

    # ===== 摄像头 =====
    def open_camera(self):
        print("\n[5] 打开 D455 摄像头...")
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 640, 480,
                                      rs.format.bgr8, 30)
            self.profile = self.pipeline.start(self.config)
            for _ in range(30):
                self.pipeline.wait_for_frames(timeout_ms=5000)
            print("  ✓ D455 就绪 (640x480 @30fps)")
            return True
        except Exception as e:
            print(f"❌ D455 失败: {e}")
            return False

    def read_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color = frames.get_color_frame()
            if not color:
                return False, None
            return True, np.asanyarray(color.get_data())
        except Exception as e:
            print(f"❌ 读帧失败: {e}")
            return False, None

    def close_camera(self):
        try:
            if hasattr(self, 'pipeline') and self.pipeline:
                self.pipeline.stop()
        except Exception:
            pass

    # ===== 核心识别 =====
    def process_frame(self, frame):
        display = frame.copy()
        faces_info = []

        try:
            faces = self.detector.get(frame)
        except Exception:
            faces = []

        for face in faces:
            bbox = face.bbox.astype(np.int32)
            kps = face.kps
            det_score = face.det_score
            arc_emb = face.normed_embedding  # InsightFace ArcFace

            # 质量过滤
            if arc_emb is None or det_score < 0.5:
                faces_info.append({
                    'bbox': bbox, 'name': 'LowQ', 'similarity': 0,
                    'matched': False, 'channel': 'filtered'
                })
                continue

            # ---- 通道1: ArcFace 相似度 ----
            arc_sims = np.dot(self.gallery_arc, arc_emb)
            arc_top1_idx = np.argmax(arc_sims)
            arc_top1_sim = arc_sims[arc_top1_idx]

            # ---- 通道2: 对齐后 AdaFace 相似度 ----
            ada_top1_sim = 0
            ada_top1_idx = 0

            if self.use_dual_channel and kps is not None:
                aligned = align_face_with_kps(frame, kps)
                if aligned is not None:
                    try:
                        ada_emb = self.recognizer.extract_feature(aligned)
                        ada_sims = np.dot(self.gallery_ada, ada_emb)
                        ada_top1_idx = np.argmax(ada_sims)
                        ada_top1_sim = ada_sims[ada_top1_idx]
                    except Exception:
                        ada_top1_sim = 0

            # ---- 融合得分 ----
            if self.use_dual_channel:
                fused_sim = (self.arcface_weight * arc_top1_sim +
                             self.adaface_weight * ada_top1_sim)

                # 两通道一致性检查
                channel_agree = (arc_top1_idx == ada_top1_idx)
                if channel_agree:
                    # 一致时加分
                    fused_sim = fused_sim * 1.05  # 5% bonus
            else:
                fused_sim = arc_top1_sim
                channel_agree = True

            # ---- 自适应阈值: 考虑 top1/top2 差距 ----
            if len(arc_sims) >= 2:
                arc_sims_sorted = np.sort(arc_sims)[::-1]
                margin = arc_sims_sorted[0] - arc_sims_sorted[1]
            else:
                margin = 1.0

            # 动态阈值: margin 大时降低阈值，margin 小时提高阈值
            adaptive_threshold = self.threshold
            if margin > 0.3:
                adaptive_threshold = self.threshold * 0.85  # 差距大，更自信
            elif margin < 0.1:
                adaptive_threshold = self.threshold * 1.2   # 差距小，更谨慎

            matched = fused_sim >= adaptive_threshold

            if matched:
                name = str(self.gallery_names[arc_top1_idx])
                if not channel_agree and self.use_dual_channel:
                    name += "?"  # 双通道不一致标记
            else:
                name = "Unknown"

            if fused_sim >= adaptive_threshold * 0.7 and not matched:
                name = "Unknown?"  # 接近阈值但不确定

            # 选择用于显示的通道标签
            channel = "dual" if self.use_dual_channel else "arc"

            faces_info.append({
                'bbox': bbox,
                'name': name,
                'similarity': float(fused_sim),
                'matched': matched,
                'channel': channel,
                'arc_sim': float(arc_top1_sim),
                'ada_sim': float(ada_top1_sim) if self.use_dual_channel else 0,
                'margin': float(margin),
                'det_score': float(det_score)
            })

        # ---- 绘制 ----
        for info in faces_info:
            bbox = info['bbox']
            color = (0, 255, 0) if info['matched'] else (0, 0, 255)
            if info['name'].endswith("?"):
                color = (0, 200, 255)

            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # 人名 (中文兼容)
            label = clean_name(info['name'])
            put_chinese_text_bg(display, label,
                                (bbox[0] + 3, bbox[1] - 22), 18,
                                (255, 255, 255), color)

            # 得分 (ASCII)
            if info['matched'] or info['name'] != "Unknown":
                detail = (f"S={info['similarity']:.2f} "
                          f"A={info.get('arc_sim', 0):.2f} "
                          f"D={info.get('ada_sim', 0):.2f}")
                cv2.putText(display, detail,
                            (bbox[0] + 5, bbox[1] + bbox[3] - bbox[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # ---- FPS / 状态 ----
        self.frame_count += 1
        elapsed = time.time() - self.fps_start
        if elapsed > 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start = time.time()

        fps_c = (0, 255, 0) if self.current_fps > 15 else (0, 200, 255)
        cv2.putText(display, f"FPS:{self.current_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_c, 2)
        cv2.putText(display, f"Th:{self.threshold:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        mode_str = "Dual" if self.use_dual_channel else "ArcFace"
        put_chinese_text(display, f"Mode:{mode_str}",
                         (10, 80), 16, (180, 180, 180))

        hints = ["Q:Quit", "T:Th+", "G:Th-", "S:Save"]
        for i, h in enumerate(hints):
            cv2.putText(display, h,
                        (display.shape[1] - 110, display.shape[0] - 20 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

        return display, faces_info

    def run(self):
        if not self.open_camera():
            return

        print("\n[6] Real-time recognition running...")
        print("  Q/ESC:Quit  T:Th+  G:Th-  S:Save")

        save_count = 0
        save_dir = os.path.join(project_root, 'temp_results')
        os.makedirs(save_dir, exist_ok=True)

        try:
            while True:
                ret, frame = self.read_frame()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

                display, faces_info = self.process_frame(frame)
                cv2.imshow('Face Recognition v2 - D455 (Aligned)', display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('t'), ord('T')):
                    self.threshold = min(1.0, self.threshold + 0.05)
                    print(f"  阈值: {self.threshold:.2f}")
                elif key in (ord('g'), ord('G')):
                    self.threshold = max(0.0, self.threshold - 0.05)
                    print(f"  阈值: {self.threshold:.2f}")
                elif key in (ord('s'), ord('S')):
                    save_count += 1
                    sp = os.path.join(save_dir, f"recog_v2_{save_count:03d}.jpg")
                    ok, buf = cv2.imencode(".jpg", display)
                    if ok:
                        with open(sp, 'wb') as f:
                            f.write(buf.tobytes())
                        print(f"  ✓ 截图: {sp}")

        finally:
            self.close_camera()
            cv2.destroyAllWindows()
            print(f"\n结束。共保存 {save_count} 张截图")


if __name__ == '__main__':
    gallery_dir = os.path.join(project_root, 'galleries')
    model_path = os.path.join(project_root, 'models', 'adaface_ir50_ms1mv2.ckpt')

    # 检查特征库
    arc = os.path.join(gallery_dir, 'gallery_arcface.npz')
    old = os.path.join(gallery_dir, 'gallery_features_realtime.npz')
    if not os.path.exists(arc) and not os.path.exists(old):
        print("❌ 无特征库，请先运行 extract_gallery_features_v2.py")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        sys.exit(1)

    system = RealTimeFaceRecognitionV2(
        gallery_dir=gallery_dir,
        model_path=model_path,
        threshold=0.45,
        arcface_weight=0.65
    )
    system.run()

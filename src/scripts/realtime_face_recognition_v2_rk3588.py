#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鲁班猫 BTB / Ubuntu 20.04 / RK3588 实时人脸识别程序
====================================================
模型文件：
  models/det_10g.rknn
  models/w600k_r50.rknn
  models/adaface_ir50.rknn

功能：
  1. Intel RealSense D455 取彩色图像
  2. det_10g.rknn 人脸检测 + 5 点关键点
  3. 5 点仿射对齐到 112x112
  4. w600k_r50.rknn 提取 ArcFace 特征
  5. adaface_ir50.rknn 提取 AdaFace 特征
  6. 双通道相似度融合识别

依赖：
  pip3 install numpy opencv-python pillow pyrealsense2
  安装板端 rknn-toolkit-lite2 后应可 import rknnlite.api

推荐目录：
  project/
    lubancat_btb_rknn_face_recognition.py
    models/
      det_10g.rknn
      w600k_r50.rknn
      adaface_ir50.rknn
    galleries/
      gallery_arcface.npz
      gallery_adaface.npz
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import pyrealsense2 as rs
except Exception as e:
    rs = None
    print("[WARN] pyrealsense2 导入失败，D455 模式不可用:", e)

try:
    from rknnlite.api import RKNNLite
except Exception as e:
    RKNNLite = None
    print("[ERROR] rknnlite.api 导入失败，请在鲁班猫板端安装 rknn-toolkit-lite2:", e)


# =========================
# 基础配置
# =========================
ARCFACE_5POINT_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)

DET_SIZE = (640, 640)      # det_10g 输入
REC_SIZE = (112, 112)      # ArcFace / AdaFace 输入


@dataclass
class FaceResult:
    bbox: np.ndarray
    kps: np.ndarray
    score: float


# =========================
# RKNN 封装
# =========================
class RKNNModel:
    def __init__(self, model_path, core_mask=None, name="model"):
        if RKNNLite is None:
            raise RuntimeError("未找到 rknnlite.api，请确认已安装 rknn-toolkit-lite2")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.model_path = model_path
        self.name = name
        self.rknn = RKNNLite()
        print(f"[RKNN] 加载 {name}: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn 失败: {model_path}, ret={ret}")
        if core_mask is None:
            core_mask = RKNNLite.NPU_CORE_AUTO
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"init_runtime 失败: {model_path}, ret={ret}")

    def infer(self, input_data):
        return self.rknn.inference(inputs=[input_data])

    def release(self):
        try:
            self.rknn.release()
        except Exception:
            pass


# =========================
# 中文显示
# =========================
def get_chinese_font(size):
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def put_chinese_text(img_bgr, text, pos, font_size, color_bgr):
    """中文/英文文字绘制，兼容 Ubuntu 字体。"""
    if not text:
        return
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = get_chinese_font(font_size)
    fg = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos, text, font=font, fill=fg)
    img_bgr[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def put_chinese_text_bg(img_bgr, text, pos, font_size, color_bgr, bg_bgr):
    """带背景的中文/英文文字绘制，保留原程序标签显示效果。"""
    if not text:
        return
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = get_chinese_font(font_size)
    fg = (color_bgr[2], color_bgr[1], color_bgr[0])
    bg = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    x, y = int(pos[0]), int(pos[1])
    draw.rectangle([x - 2, y, x + tw + 6, y + th + 5], fill=bg)
    draw.text((x + 2, y + 1), text, font=font, fill=fg)
    img_bgr[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# 兼容旧函数名
put_text_bg = put_chinese_text_bg


def clean_name(name):
    s = str(name)
    return s[7:] if s.startswith("Person_") else s


def l2_normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(x)
    return x / max(n, eps)


# =========================
# 图像预处理 / 几何解码
# =========================
def letterbox_bgr(img_bgr, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img_bgr.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    resized_w, resized_h = int(round(w * r)), int(round(h * r))
    dw = (new_w - resized_w) / 2
    dh = (new_h - resized_h) / 2

    resized = cv2.resize(img_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, left, top


def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(boxes, scores, thresh=0.45):
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def align_face_with_kps(img_bgr, kps):
    if kps is None or np.asarray(kps).shape[0] < 5:
        return None
    kps = np.asarray(kps, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(
        kps, ARCFACE_5POINT_TEMPLATE,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )
    if M is None:
        return None
    return cv2.warpAffine(img_bgr, M, REC_SIZE, borderMode=cv2.BORDER_REPLICATE)


# =========================
# SCRFD 检测器 RKNN 后处理
# =========================
class SCRFDRKNNDetector:
    def __init__(self, model_path, score_thresh=0.5, nms_thresh=0.45):
        self.model = RKNNModel(model_path, name="det_10g")
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.input_size = DET_SIZE
        self.strides = [8, 16, 32]
        self.num_anchors = 2

    @staticmethod
    def _to_2d(arr):
        arr = np.asarray(arr)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        return arr.astype(np.float32)

    def _split_scrfd_outputs(self, outputs):
        """
        兼容常见 SCRFD 输出：
        9 输出：score8, score16, score32, bbox8, bbox16, bbox32, kps8, kps16, kps32
        部分转换会改变顺序，这里优先按照最后一维 1/4/10 自动归类。
        """
        scores, bboxes, kpss = [], [], []
        for out in outputs:
            arr = self._to_2d(out)
            c = arr.shape[1]
            if c == 1:
                scores.append(arr)
            elif c == 4:
                bboxes.append(arr)
            elif c == 10:
                kpss.append(arr)
        if len(scores) == 3 and len(bboxes) == 3 and len(kpss) == 3:
            # 按候选点数量从多到少对应 stride 8/16/32
            scores = sorted(scores, key=lambda x: x.shape[0], reverse=True)
            bboxes = sorted(bboxes, key=lambda x: x.shape[0], reverse=True)
            kpss = sorted(kpss, key=lambda x: x.shape[0], reverse=True)
            return scores, bboxes, kpss
        if len(outputs) >= 9:
            return ([self._to_2d(outputs[i]) for i in range(3)],
                    [self._to_2d(outputs[i]) for i in range(3, 6)],
                    [self._to_2d(outputs[i]) for i in range(6, 9)])
        raise RuntimeError(f"det_10g 输出数量/形状不符合 SCRFD 预期: {[np.asarray(o).shape for o in outputs]}")

    def detect(self, img_bgr, max_num=0):
        padded, scale, pad_x, pad_y = letterbox_bgr(img_bgr, self.input_size)

        # 默认假设 RKNN 转换时已经在 config 设置 mean/std，所以板端输入 uint8 RGB NHWC。
        inp = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(inp, axis=0).astype(np.uint8)

        outputs = self.model.infer(inp)
        score_list, bbox_list, kps_list = self._split_scrfd_outputs(outputs)

        input_w, input_h = self.input_size
        all_boxes, all_scores, all_kps = [], [], []

        for idx, stride in enumerate(self.strides):
            scores = score_list[idx].reshape(-1)
            bbox_pred = bbox_list[idx]
            kps_pred = kps_list[idx]

            feat_h = input_h // stride
            feat_w = input_w // stride
            anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape(-1, 2)
            if self.num_anchors > 1:
                anchor_centers = np.repeat(anchor_centers, self.num_anchors, axis=0)

            # 有些模型输出的候选点数量可能已经不含 num_anchors，做一次兼容。
            n = min(scores.shape[0], bbox_pred.shape[0], kps_pred.shape[0], anchor_centers.shape[0])
            scores = scores[:n]
            bbox_pred = bbox_pred[:n]
            kps_pred = kps_pred[:n]
            anchor_centers = anchor_centers[:n]

            pos_inds = np.where(scores >= self.score_thresh)[0]
            if pos_inds.size == 0:
                continue
            boxes = distance2bbox(anchor_centers[pos_inds], bbox_pred[pos_inds] * stride)
            kps = distance2kps(anchor_centers[pos_inds], kps_pred[pos_inds] * stride).reshape(-1, 5, 2)

            # 映射回原图坐标
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
            kps[:, :, 0] = (kps[:, :, 0] - pad_x) / scale
            kps[:, :, 1] = (kps[:, :, 1] - pad_y) / scale

            h0, w0 = img_bgr.shape[:2]
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0 - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0 - 1)

            all_boxes.append(boxes)
            all_scores.append(scores[pos_inds])
            all_kps.append(kps)

        if not all_boxes:
            return []

        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        kpss = np.vstack(all_kps)
        keep = nms(boxes, scores, self.nms_thresh)
        if max_num > 0:
            keep = keep[:max_num]

        faces = []
        for i in keep:
            faces.append(FaceResult(boxes[i].astype(np.float32), kpss[i].astype(np.float32), float(scores[i])))
        return faces

    def release(self):
        self.model.release()


# =========================
# 识别模型 RKNN
# =========================
class FaceRecognizerRKNN:
    def __init__(self, arcface_path, adaface_path=None):
        self.arc_model = RKNNModel(arcface_path, name="w600k_r50")
        self.ada_model = RKNNModel(adaface_path, name="adaface_ir50") if adaface_path else None

    @staticmethod
    def preprocess_face(img_bgr):
        """
        默认假设 RKNN 转换时已经配置 mean/std：
        ArcFace / AdaFace 输入 RGB uint8 NHWC，归一化交给 RKNN 内部处理。
        """
        img = cv2.resize(img_bgr, REC_SIZE, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(img, axis=0).astype(np.uint8)

    def extract_arc(self, aligned_bgr):
        inp = self.preprocess_face(aligned_bgr)
        outs = self.arc_model.infer(inp)
        return l2_normalize(outs[0])

    def extract_ada(self, aligned_bgr):
        if self.ada_model is None:
            return None
        inp = self.preprocess_face(aligned_bgr)
        outs = self.ada_model.infer(inp)
        return l2_normalize(outs[0])

    def release(self):
        self.arc_model.release()
        if self.ada_model:
            self.ada_model.release()


# =========================
# D455 摄像头
# =========================
class D455Camera:
    def __init__(self, width=640, height=480, fps=30):
        if rs is None:
            raise RuntimeError("pyrealsense2 不可用")
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None

    def open(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.pipeline.start(config)
        for _ in range(15):
            self.pipeline.wait_for_frames(timeout_ms=5000)
        print(f"[D455] 打开成功: {self.width}x{self.height}@{self.fps}")

    def read(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        color = frames.get_color_frame()
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data())

    def close(self):
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None


# =========================
# 主系统
# =========================
class RKNNFaceRecognitionSystem:
    def __init__(self, args):
        print("=" * 70)
        print("鲁班猫 BTB 实时人脸识别系统 v2-RKNN")
        print("保留功能：D455 + 5点对齐 + ArcFace/AdaFace 双通道融合 + 自适应阈值")
        print("=" * 70)
        self.args = args
        self.threshold = args.threshold
        self.arcface_weight = args.arcface_weight
        self.adaface_weight = 1.0 - self.arcface_weight

        det_path = os.path.join(args.model_dir, "det_10g.rknn")
        arc_path = os.path.join(args.model_dir, "w600k_r50.rknn")
        ada_path = os.path.join(args.model_dir, "adaface_ir50.rknn")

        self.detector = SCRFDRKNNDetector(det_path, args.det_thresh, args.nms_thresh)
        self.recognizer = FaceRecognizerRKNN(arc_path, ada_path if os.path.exists(ada_path) else None)

        self._load_gallery(args.gallery_dir)

        self.frame_count = 0
        self.fps_start = time.time()
        self.current_fps = 0.0

    def _load_gallery(self, gallery_dir):
        arc_path = os.path.join(gallery_dir, "gallery_arcface.npz")
        ada_path = os.path.join(gallery_dir, "gallery_adaface.npz")
        old_path = os.path.join(gallery_dir, "gallery_features_realtime.npz")

        if os.path.exists(arc_path):
            arc_data = np.load(arc_path, allow_pickle=True)
            self.gallery_arc = np.asarray(arc_data["features"], dtype=np.float32)
            self.gallery_arc = self.gallery_arc / np.maximum(np.linalg.norm(self.gallery_arc, axis=1, keepdims=True), 1e-12)
            self.gallery_names = arc_data["names"]
            self.gallery_ids = arc_data["person_ids"] if "person_ids" in arc_data else np.arange(len(self.gallery_names))
        elif os.path.exists(old_path):
            data = np.load(old_path, allow_pickle=True)
            self.gallery_arc = np.asarray(data["features"], dtype=np.float32)
            self.gallery_arc = self.gallery_arc / np.maximum(np.linalg.norm(self.gallery_arc, axis=1, keepdims=True), 1e-12)
            self.gallery_names = data["names"]
            self.gallery_ids = data["person_ids"] if "person_ids" in data else np.arange(len(self.gallery_names))
        else:
            raise FileNotFoundError(f"未找到特征库: {arc_path} 或 {old_path}")

        self.use_dual_channel = False
        self.gallery_ada = None
        if os.path.exists(ada_path) and self.recognizer.ada_model is not None:
            ada_data = np.load(ada_path, allow_pickle=True)
            self.gallery_ada = np.asarray(ada_data["features"], dtype=np.float32)
            self.gallery_ada = self.gallery_ada / np.maximum(np.linalg.norm(self.gallery_ada, axis=1, keepdims=True), 1e-12)
            self.use_dual_channel = len(self.gallery_ada) == len(self.gallery_arc)

        print(f"[Gallery] 已加载 {len(self.gallery_names)} 个特征")
        print(f"[Gallery] 模式: {'ArcFace + AdaFace 双通道' if self.use_dual_channel else 'ArcFace 单通道'}")

    def process_frame(self, frame_bgr):
        display = frame_bgr.copy()
        faces = self.detector.detect(frame_bgr, max_num=self.args.max_faces)
        infos = []

        for face in faces:
            bbox = face.bbox.astype(np.int32)
            aligned = align_face_with_kps(frame_bgr, face.kps)
            if aligned is None:
                continue

            arc_emb = self.recognizer.extract_arc(aligned)
            arc_sims = np.dot(self.gallery_arc, arc_emb)
            arc_idx = int(np.argmax(arc_sims))
            arc_sim = float(arc_sims[arc_idx])

            ada_idx = arc_idx
            ada_sim = 0.0
            channel_agree = True
            if self.use_dual_channel:
                ada_emb = self.recognizer.extract_ada(aligned)
                ada_sims = np.dot(self.gallery_ada, ada_emb)
                ada_idx = int(np.argmax(ada_sims))
                ada_sim = float(ada_sims[ada_idx])
                channel_agree = arc_idx == ada_idx
                fused = self.arcface_weight * arc_sim + self.adaface_weight * ada_sim
                if channel_agree:
                    fused *= 1.05
            else:
                fused = arc_sim

            if len(arc_sims) >= 2:
                sorted_sims = np.sort(arc_sims)[::-1]
                margin = float(sorted_sims[0] - sorted_sims[1])
            else:
                margin = 1.0

            adaptive_th = self.threshold
            if margin > 0.30:
                adaptive_th *= 0.85
            elif margin < 0.10:
                adaptive_th *= 1.20

            matched = fused >= adaptive_th
            name = str(self.gallery_names[arc_idx]) if matched else "Unknown"
            if matched and self.use_dual_channel and not channel_agree:
                name += "?"
            if not matched and fused >= adaptive_th * 0.7:
                name = "Unknown?"

            infos.append({
                "bbox": bbox,
                "name": name,
                "score": float(fused),
                "arc_sim": arc_sim,
                "ada_sim": ada_sim,
                "det_score": face.score,
                "matched": matched,
            })

        for info in infos:
            x1, y1, x2, y2 = info["bbox"]
            color = (0, 255, 0) if info["matched"] else (0, 0, 255)
            if str(info["name"]).endswith("?"):
                color = (0, 200, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            put_chinese_text_bg(display, clean_name(info["name"]), (x1 + 3, max(0, y1 - 24)), 18, (255, 255, 255), color)
            if info["matched"] or info["name"] != "Unknown":
                detail = f"S={info['score']:.2f} A={info['arc_sim']:.2f} D={info['ada_sim']:.2f}"
                cv2.putText(display, detail, (x1 + 5, min(display.shape[0] - 8, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # FPS / 状态栏：保留原程序显示信息
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
        put_chinese_text(display, f"Mode:{mode_str}", (10, 80), 16, (180, 180, 180))

        hints = ["Q:Quit", "T:Th+", "G:Th-", "S:Save"]
        for i, h in enumerate(hints):
            cv2.putText(display, h,
                        (display.shape[1] - 110, display.shape[0] - 20 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
        return display, infos

    def run(self):
        cam = D455Camera(self.args.cam_width, self.args.cam_height, self.args.cam_fps)
        cam.open()
        save_dir = self.args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        save_count = 0
        print("[RUN] Q/ESC 退出，T 增大阈值，G 减小阈值，S 保存截图")
        try:
            while True:
                ok, frame = cam.read()
                if not ok or frame is None:
                    continue
                display, _ = self.process_frame(frame)
                if not self.args.no_display:
                    cv2.imshow("RKNN Face Recognition - LubanCat BTB", display)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = 255
                if key in (ord('q'), ord('Q'), 27):
                    break
                if key in (ord('t'), ord('T')):
                    self.threshold = min(1.0, self.threshold + 0.05)
                    print("阈值:", self.threshold)
                if key in (ord('g'), ord('G')):
                    self.threshold = max(0.0, self.threshold - 0.05)
                    print("阈值:", self.threshold)
                if key in (ord('s'), ord('S')):
                    save_count += 1
                    path = os.path.join(save_dir, f"rknn_recog_{save_count:03d}.jpg")
                    ok, buf = cv2.imencode(".jpg", display)
                    if ok:
                        with open(path, "wb") as f:
                            f.write(buf.tobytes())
                        print("  ✓ 截图:", path)
                    else:
                        print("  ✗ 截图保存失败:", path)
        finally:
            cam.close()
            cv2.destroyAllWindows()
            self.release()

    def release(self):
        self.detector.release()
        self.recognizer.release()


def parse_args():
    parser = argparse.ArgumentParser(description="鲁班猫 BTB RKNN D455 实时人脸识别")
    
    # 修复路径：脚本在 src/scripts/ 下，需要定位到项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))   # .../2d3dadvface/src/scripts/
    src_dir = os.path.dirname(script_dir)                      # .../2d3dadvface/src/
    project_root = os.path.dirname(src_dir)                    # .../2d3dadvface/2d3dadvface/
    
    default_model_dir = os.path.join(project_root, "models")
    default_gallery_dir = os.path.join(project_root, "galleries")
    default_save_dir = os.path.join(project_root, "temp_results")
    
    print(f"[路径] 脚本目录: {script_dir}")
    print(f"[路径] 项目根目录: {project_root}")
    print(f"[路径] 模型目录: {default_model_dir}")
    print(f"[路径] 特征库目录: {default_gallery_dir}")
    
    parser.add_argument("--model-dir", default=default_model_dir, help="RKNN 模型目录")
    parser.add_argument("--gallery-dir", default=default_gallery_dir, help="特征库目录")
    parser.add_argument("--threshold", type=float, default=0.45, help="基础识别阈值")
    parser.add_argument("--arcface-weight", type=float, default=0.65, help="ArcFace 融合权重")
    parser.add_argument("--det-thresh", type=float, default=0.5, help="人脸检测阈值")
    parser.add_argument("--nms-thresh", type=float, default=0.45, help="NMS 阈值")
    parser.add_argument("--max-faces", type=int, default=0, help="最大检测人脸数，0 表示不限制")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--save-dir", default=default_save_dir)
    parser.add_argument("--no-display", action="store_true", help="无窗口运行")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    system = RKNNFaceRecognitionSystem(args)
    system.run()

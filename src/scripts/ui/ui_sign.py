"""
人脸签到系统 - 基于 Intel RealSense D455 + AdaFace
产品级 PyQt5 界面：左侧人员列表 + 右侧实时识别画面
"""
import os
import sys
import json
import time
import threading
from datetime import datetime
from collections import OrderedDict, deque

# 项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import pyrealsense2 as rs
import insightface
from insightface.app import FaceAnalysis

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QFrame,
    QSlider, QSizePolicy, QSpacerItem, QProgressBar
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QRect, QPropertyAnimation,
    QEasingCurve, QPoint
)
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QPainter, QBrush,
    QLinearGradient, QFontDatabase, QIcon, QPen, QPainterPath
)

from src.utils.adaface_recognizer import AdaFaceRecognizer

# ─── 全局样式常量 ─────────────────────────────────────────────
COLOR_BG_DARK = "#0a0e14"
COLOR_BG_PANEL = "#131820"
COLOR_BG_CARD = "#1a1f2b"
COLOR_BG_CARD_HOVER = "#212837"
COLOR_ACCENT = "#00d4aa"
COLOR_ACCENT_DIM = "#008866"
COLOR_SUCCESS = "#00e676"
COLOR_WARNING = "#ffb300"
COLOR_DANGER = "#ff5252"
COLOR_TEXT_PRIMARY = "#e0e0e0"
COLOR_TEXT_SECONDARY = "#808a9a"
COLOR_TEXT_MUTED = "#4a5568"
COLOR_BORDER = "#252a36"
COLOR_DIVIDER = "#1e2430"
COLOR_SIGNED_BG = "rgba(0, 212, 170, 0.12)"
COLOR_UNSIGNED_BG = "rgba(255, 255, 255, 0.03)"

FONT_FAMILY = "Microsoft YaHei, Segoe UI, sans-serif"

STYLE_GLOBAL = f"""
QMainWindow {{
    background-color: {COLOR_BG_DARK};
}}
QWidget {{
    font-family: "{FONT_FAMILY}";
    color: {COLOR_TEXT_PRIMARY};
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: {COLOR_BG_PANEL};
    width: 6px;
    border-radius: 3px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {COLOR_BORDER};
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLOR_TEXT_MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {COLOR_BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {COLOR_ACCENT};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {COLOR_SUCCESS};
}}
QSlider::sub-page:horizontal {{
    background: {COLOR_ACCENT};
    border-radius: 2px;
}}
"""

# ─── 清理人名字段 ─────────────────────────────────────────────
def clean_display_name(name):
    """从 'Person_仲昭腾' 提取 '仲昭腾'"""
    if name.startswith("Person_"):
        return name[7:]
    return name


# ─── 签到追踪管理器 ───────────────────────────────────────────
class SignInManager:
    """管理签到状态，追踪签到时间和去重"""

    def __init__(self, names, person_ids):
        self._records = OrderedDict()
        for i, (name, pid) in enumerate(zip(names, person_ids)):
            display = clean_display_name(name)
            self._records[display] = {
                "index": i,
                "person_id": str(pid),
                "raw_name": str(name),
                "signed": False,
                "sign_time": None,
                "confidence": 0.0,
            }

    @property
    def records(self):
        return self._records

    def mark_signed(self, display_name, confidence):
        if display_name in self._records:
            rec = self._records[display_name]
            if not rec["signed"]:
                rec["signed"] = True
                rec["sign_time"] = datetime.now().strftime("%H:%M:%S")
                rec["confidence"] = confidence
                return True  # 新签到
            else:
                # 更新置信度
                rec["confidence"] = max(rec["confidence"], confidence)
                return False  # 已签到过
        return False

    def total_count(self):
        return len(self._records)

    def signed_count(self):
        return sum(1 for r in self._records.values() if r["signed"])

    def remaining_count(self):
        return self.total_count() - self.signed_count()

    def reset_all(self):
        for r in self._records.values():
            r["signed"] = False
            r["sign_time"] = None
            r["confidence"] = 0.0


# ─── 识别工作线程 ─────────────────────────────────────────────
class RecognitionThread(QThread):
    """后台线程：读帧 -> 检测 -> 识别"""
    frame_ready = pyqtSignal(np.ndarray, list)        # display_frame, faces_info
    sign_in_event = pyqtSignal(str, float)             # display_name, confidence
    fps_update = pyqtSignal(float)                     # fps
    error_occurred = pyqtSignal(str)                   # error message

    def __init__(self, recognizer, detector, gallery_features, gallery_names,
                 person_ids, threshold=0.30, stabilize_frames=10, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.detector = detector
        self.gallery_features = gallery_features
        self.gallery_names = gallery_names
        self.person_ids = person_ids
        self.threshold = threshold
        self.stabilize_frames = stabilize_frames  # 连续确认帧数
        self._running = False
        self._pipeline = None
        self._config = None
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()

        # 帧级一致性缓冲区: face_id -> deque of (matched_name_or_None)
        self._consensus_buffer = {}
        # 已经触发了签到事件的名单（本线程生命周期内，避免重复emit）
        self._sign_emitted = set()

    def set_threshold(self, val):
        self.threshold = val

    def reset_sign_state(self):
        """重置签到状态（清除已触发名单和共识缓冲）"""
        self._sign_emitted.clear()
        self._consensus_buffer.clear()

    def _init_camera(self):
        try:
            self._pipeline = rs.pipeline()
            self._config = rs.config()
            self._config.enable_stream(
                rs.stream.color, 640, 480, rs.format.bgr8, 30
            )
            self._pipeline.start(self._config)
            for _ in range(30):
                self._pipeline.wait_for_frames(timeout_ms=5000)
            return True
        except Exception as e:
            self.error_occurred.emit(f"摄像头初始化失败: {e}")
            return False

    def _stop_camera(self):
        try:
            if self._pipeline:
                self._pipeline.stop()
        except Exception:
            pass

    def run(self):
        if not self._init_camera():
            return
        self._running = True

        # 将 gallery_features 转为一次性计算好的数据避免在每个识别线程里误操作
        gfeat = self.gallery_features
        gnames = self.gallery_names

        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
                if frame is None or frame.size == 0:
                    continue
            except Exception:
                continue

            display_frame, faces_info = self._process(frame, gfeat, gnames)
            self.frame_ready.emit(display_frame, faces_info)

            # FPS
            self._frame_count += 1
            elapsed = time.time() - self._fps_timer
            if elapsed > 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = time.time()
                self.fps_update.emit(self._fps)

            # 帧级一致性确认后触发签到
            for info in faces_info:
                if not info["matched"]:
                    continue
                display_name = clean_display_name(info["name"])
                if display_name in self._sign_emitted:
                    continue

                bbox = info["bbox"]
                face_key = self._get_face_key(bbox)

                # 获取或创建该人脸的滑动窗口
                if face_key not in self._consensus_buffer:
                    self._consensus_buffer[face_key] = deque(maxlen=self.stabilize_frames)
                buf = self._consensus_buffer[face_key]
                buf.append(display_name)

                # 检查共识：最近N帧中同一人占比 > 80%（即至少需要 N*0.8+1 帧匹配）
                if len(buf) >= self.stabilize_frames:
                    from collections import Counter
                    counter = Counter(buf)
                    top_name, top_count = counter.most_common(1)[0]
                    min_required = self.stabilize_frames - 1  # 最多容错1帧
                    if top_name != "Unknown" and top_count >= min_required:
                        self._sign_emitted.add(top_name)
                        # 找到对应帧的相似度
                        self.sign_in_event.emit(top_name, info["similarity"])
                        # 清除该人脸缓冲，避免重复触发
                        buf.clear()

            # 定期清理不再出现的 face_key（超过 stabilize_frames*2 帧无更新的清理）
            active_keys = {self._get_face_key(f["bbox"]) for f in faces_info}
            stale_keys = [k for k in self._consensus_buffer if k not in active_keys]
            for k in stale_keys:
                del self._consensus_buffer[k]

    @staticmethod
    def _get_face_key(bbox, grid=40):
        """根据人脸 bbox 中心坐标生成跟踪 key（网格化近似追踪）"""
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        gx = cx // grid
        gy = cy // grid
        return f"{gx}_{gy}"

    def _process(self, frame, gfeat, gnames):
        display = frame.copy()
        faces_info = []
        try:
            faces = self.detector.get(frame)
        except Exception:
            faces = []

        for face in faces:
            bbox = face.bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox
            margin = 20
            x1m = max(0, x1 - margin)
            y1m = max(0, y1 - margin)
            x2m = min(frame.shape[1], x2 + margin)
            y2m = min(frame.shape[0], y2 + margin)
            face_img = frame[y1m:y2m, x1m:x2m]
            if face_img.size == 0:
                continue

            matched = False
            name = "Unknown"
            similarity = 0.0
            try:
                qfeat = self.recognizer.extract_feature(face_img)
                result = self.recognizer.recognize(
                    qfeat, gfeat, gnames, threshold=self.threshold
                )
                name = result["name"]
                similarity = result["similarity"]
                matched = result["matched"]
            except Exception:
                pass

            faces_info.append({
                "bbox": bbox,
                "name": name,
                "similarity": similarity,
                "matched": matched,
            })

            # 绘制
            color = (0, 212, 170) if matched else (255, 82, 82)  # BGR
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            label = clean_display_name(name) if matched else name
            sub = f"{similarity:.2f}"
            font_scale = 0.6
            thick = 2
            (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
            cv2.rectangle(display,
                          (bbox[0], bbox[1] - h1 - 12),
                          (bbox[0] + w1 + 12, bbox[1]),
                          color, -1)
            cv2.putText(display, label, (bbox[0] + 6, bbox[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick)

            fs_small = 0.4
            (w2, h2), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, fs_small, 1)
            cv2.rectangle(display,
                          (bbox[2] - w2 - 6, bbox[1] + 4),
                          (bbox[2], bbox[1] + h2 + 8),
                          (10, 14, 20), -1)
            cv2.putText(display, sub, (bbox[2] - w2 - 2, bbox[1] + h2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_small,
                        (0, 212, 170) if matched else (255, 82, 82), 1)

        # Canny 边缘检测叠加（半透明，增加高级感）
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 60, 150)
            edge_overlay = np.zeros_like(display)
            edge_overlay[:, :, 0] = edges  # B 通道
            edge_overlay[:, :, 1] = edges  # G 通道
            edge_overlay[:, :, 2] = edges  # R 通道
            display = cv2.addWeighted(display, 1.0, edge_overlay, 0.06, 0)
        except Exception:
            pass

        return display, faces_info

    def stop(self):
        self._running = False
        self._stop_camera()
        self.wait(3000)


# ─── 人员卡片组件 ─────────────────────────────────────────────
class PersonCard(QFrame):
    """左侧列表中每个人的卡片"""

    def __init__(self, display_name, person_id, parent=None):
        super().__init__(parent)
        self.display_name = display_name
        self.person_id = person_id
        self._signed = False
        self._init_ui()

    def _init_ui(self):
        self.setFixedHeight(56)
        self.setObjectName("personCard")
        self.setStyleSheet(f"""
            #personCard {{
                background: {COLOR_BG_CARD};
                border-radius: 8px;
                margin: 2px 8px;
                border: 1px solid {COLOR_BORDER};
            }}
            #personCard:hover {{
                background: {COLOR_BG_CARD_HOVER};
                border-color: {COLOR_TEXT_MUTED};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 14, 6)
        layout.setSpacing(10)

        # 头像占位（圆形）
        self.avatar = QLabel()
        self.avatar.setFixedSize(40, 40)
        self.avatar.setAlignment(Qt.AlignCenter)
        self._update_avatar()
        layout.addWidget(self.avatar)

        # 名字 + ID
        info_layout = QVBoxLayout()
        info_layout.setSpacing(1)

        self.name_label = QLabel(self.display_name)
        self.name_label.setStyleSheet(f"""
            font-size: 14px; font-weight: 600; color: {COLOR_TEXT_PRIMARY};
            background: transparent; border: none;
        """)
        info_layout.addWidget(self.name_label)

        self.id_label = QLabel(f"ID: {self.person_id}")
        self.id_label.setStyleSheet(f"""
            font-size: 11px; color: {COLOR_TEXT_MUTED};
            background: transparent; border: none;
        """)
        info_layout.addWidget(self.id_label)

        layout.addLayout(info_layout)
        layout.addStretch()

        # 签到状态
        self.status_label = QLabel("未签到")
        self.status_label.setFixedWidth(60)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"""
            font-size: 11px; font-weight: bold; color: {COLOR_TEXT_MUTED};
            background: transparent; border: none;
        """)
        layout.addWidget(self.status_label)

        # 签到图标
        self.check_icon = QLabel("○")
        self.check_icon.setFixedSize(22, 22)
        self.check_icon.setAlignment(Qt.AlignCenter)
        self.check_icon.setStyleSheet(f"""
            font-size: 14px; color: {COLOR_TEXT_MUTED};
            background: transparent; border: none;
        """)
        layout.addWidget(self.check_icon)

    def _update_avatar(self):
        """生成圆形头像占位（首字）"""
        initial = self.display_name[0] if self.display_name else "?"
        # 用颜色哈希生成一致的背景色
        hue = sum(ord(c) for c in self.display_name) % 360
        r, g, b = self._hsl_to_rgb(hue / 360.0, 0.45, 0.35)
        bg = f"rgb({r},{g},{b})"
        self.avatar.setStyleSheet(
            f"background: {bg}; border-radius: 20px; font-size: 16px; "
            f"font-weight: bold; color: white;"
        )
        self.avatar.setText(initial)

    @staticmethod
    def _hsl_to_rgb(h, s, l):
        """HSL -> RGB (0-255)"""
        import colorsys
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return int(r * 255), int(g * 255), int(b * 255)

    def set_signed(self, signed, sign_time=None, confidence=0.0):
        self._signed = signed
        if signed:
            self.setStyleSheet(f"""
                #personCard {{
                    background: {COLOR_SIGNED_BG};
                    border-radius: 8px;
                    margin: 2px 8px;
                    border: 1px solid rgba(0, 212, 170, 0.25);
                }}
            """)
            self.status_label.setText("已签到")
            self.status_label.setStyleSheet(f"""
                font-size: 11px; font-weight: bold; color: {COLOR_ACCENT};
                background: transparent; border: none;
            """)
            self.check_icon.setText("✓")
            self.check_icon.setStyleSheet(f"""
                font-size: 14px; color: {COLOR_ACCENT};
                background: transparent; border: none;
            """)
        else:
            self.setStyleSheet(f"""
                #personCard {{
                    background: {COLOR_BG_CARD};
                    border-radius: 8px;
                    margin: 2px 8px;
                    border: 1px solid {COLOR_BORDER};
                }}
            """)
            self.status_label.setText("未签到")
            self.status_label.setStyleSheet(f"""
                font-size: 11px; font-weight: bold; color: {COLOR_TEXT_MUTED};
                background: transparent; border: none;
            """)
            self.check_icon.setText("○")
            self.check_icon.setStyleSheet(f"""
                font-size: 14px; color: {COLOR_TEXT_MUTED};
                background: transparent; border: none;
            """)


# ─── 统计面板 ─────────────────────────────────────────────────
class StatsPanel(QFrame):
    """顶部统计信息：总数 / 已签到 / 剩余"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(100)
        self.setObjectName("statsPanel")
        self.setStyleSheet(f"""
            #statsPanel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLOR_BG_PANEL}, stop:1 {COLOR_BG_CARD});
                border-radius: 10px;
                margin: 12px 12px 4px 12px;
                border: 1px solid {COLOR_BORDER};
            }}
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(0)

        self.total_widget = self._make_stat("总数", "0", COLOR_TEXT_PRIMARY)
        layout.addWidget(self.total_widget)
        layout.addStretch()

        self.signed_widget = self._make_stat("已签到", "0", COLOR_ACCENT)
        layout.addWidget(self.signed_widget)
        layout.addStretch()

        self.remaining_widget = self._make_stat("待签到", "0", COLOR_WARNING)
        layout.addWidget(self.remaining_widget)
        layout.addStretch()

        # 进度条
        self.progress = QProgressBar()
        self.progress.setFixedSize(80, 4)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background: {COLOR_BORDER};
                border-radius: 2px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: {COLOR_ACCENT};
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self.progress)

    def _make_stat(self, label, value, color):
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(2)

        val_lbl = QLabel(value)
        val_lbl.setObjectName("statValue")
        val_lbl.setStyleSheet(f"""
            font-size: 28px; font-weight: 700; color: {color};
            background: transparent; border: none;
        """)
        val_lbl.setAlignment(Qt.AlignCenter)
        vl.addWidget(val_lbl)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"""
            font-size: 11px; color: {COLOR_TEXT_SECONDARY};
            background: transparent; border: none;
        """)
        lbl.setAlignment(Qt.AlignCenter)
        vl.addWidget(lbl)

        return w

    def update_stats(self, total, signed, remaining):
        layout = self.layout()
        # total
        tw = layout.itemAt(0).widget()
        tw.findChild(QLabel, "statValue").setText(str(total))
        # signed
        sw = layout.itemAt(2).widget()
        sw.findChild(QLabel, "statValue").setText(str(signed))
        # remaining
        rw = layout.itemAt(4).widget()
        rw.findChild(QLabel, "statValue").setText(str(remaining))
        # progress
        pct = int(signed / max(total, 1) * 100)
        self.progress.setValue(pct)


# ─── 主窗口 ───────────────────────────────────────────────────
class SignInWindow(QMainWindow):
    """人脸签到系统主窗口"""

    def __init__(self, gallery_path, model_path, threshold=0.30):
        super().__init__()
        self.gallery_path = gallery_path
        self.model_path = model_path
        self.threshold = threshold

        self._check_files()
        self._load_gallery()
        self._init_models()
        self._init_ui()
        self._init_thread()

    # ── 文件检查 ─────────────────────────────────────────────
    def _check_files(self):
        if not os.path.exists(self.gallery_path):
            msg = f"特征库不存在: {self.gallery_path}\n请先运行 extract_gallery_features_realtime.py"
            QMessageBox = __import__("PyQt5.QtWidgets", fromlist=["QMessageBox"]).QMessageBox
            QMessageBox.critical(None, "错误", msg)
            sys.exit(1)
        if not os.path.exists(self.model_path):
            msg = f"模型文件不存在: {self.model_path}"
            QMessageBox = __import__("PyQt5.QtWidgets", fromlist=["QMessageBox"]).QMessageBox
            QMessageBox.critical(None, "错误", msg)
            sys.exit(1)

    # ── 加载特征库 ───────────────────────────────────────────
    def _load_gallery(self):
        data = np.load(self.gallery_path, allow_pickle=True)
        self.gallery_features = data["features"]
        self.gallery_names = data["names"]
        self.gallery_person_ids = data["person_ids"]
        self.sign_manager = SignInManager(self.gallery_names, self.gallery_person_ids)

    # ── 初始化识别模型 ───────────────────────────────────────
    def _init_models(self):
        self.recognizer = AdaFaceRecognizer(
            model_path=self.model_path, device="cuda"
        )
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            self.detector = FaceAnalysis(
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        else:
            self.detector = FaceAnalysis(providers=["CPUExecutionProvider"])
        self.detector.prepare(ctx_id=0, det_size=(640, 640))

    # ── 初始化界面 ───────────────────────────────────────────
    def _init_ui(self):
        self.setWindowTitle("人脸签到系统 · FaceSign")
        self.setMinimumSize(1200, 750)
        self.resize(1360, 820)
        self.setStyleSheet(STYLE_GLOBAL)

        # 中央容器
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ── 左侧面板（人员列表） ─────────────────────────
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 3)

        # ── 右侧面板（视频 + 控制） ─────────────────────
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 7)

        # 应用黑暗标题栏颜色（Windows）
        self._apply_dark_titlebar()

    def _apply_dark_titlebar(self):
        """尽量让 Windows 标题栏变为深色"""
        try:
            import ctypes
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            hwnd = int(self.winId())
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(ctypes.c_int(1)),
                ctypes.sizeof(ctypes.c_int)
            )
        except Exception:
            pass

    # ── 左侧面板 ─────────────────────────────────────────────
    def _create_left_panel(self):
        panel = QFrame()
        panel.setObjectName("leftPanel")
        panel.setStyleSheet(f"""
            #leftPanel {{
                background: {COLOR_BG_PANEL};
                border-radius: 12px;
                border: 1px solid {COLOR_BORDER};
            }}
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = self._create_left_header()
        layout.addWidget(header)

        # 分割线
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {COLOR_DIVIDER}; border: none;")
        layout.addWidget(div)

        # 统计面板
        self.stats_panel = StatsPanel()
        layout.addWidget(self.stats_panel)

        # 分割线
        div2 = QFrame()
        div2.setFixedHeight(1)
        div2.setStyleSheet(f"background: {COLOR_DIVIDER}; border: none; margin: 0 12px;")
        layout.addWidget(div2)

        # 人员列表标题
        list_header = QLabel("  人员列表")
        list_header.setFixedHeight(32)
        list_header.setStyleSheet(f"""
            font-size: 12px; font-weight: 600; color: {COLOR_TEXT_SECONDARY};
            background: transparent; border: none; padding: 6px 12px;
        """)
        layout.addWidget(list_header)

        # 可滚动人员列表
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QWidget#personListWidget {{ background: transparent; }}
        """)

        self.person_list_widget = QWidget()
        self.person_list_widget.setObjectName("personListWidget")
        self.person_list_layout = QVBoxLayout(self.person_list_widget)
        self.person_list_layout.setContentsMargins(0, 4, 0, 12)
        self.person_list_layout.setSpacing(2)
        self.person_list_layout.addStretch()

        self.scroll_area.setWidget(self.person_list_widget)
        layout.addWidget(self.scroll_area, 1)

        # 底部按钮
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(12, 8, 12, 12)
        btn_layout.setSpacing(8)

        self.btn_reset = self._make_button("重置签到", COLOR_TEXT_SECONDARY, COLOR_BORDER)
        self.btn_reset.clicked.connect(self._on_reset)
        btn_layout.addWidget(self.btn_reset)

        self.btn_export = self._make_button("导出记录", COLOR_TEXT_SECONDARY, COLOR_BORDER)
        self.btn_export.clicked.connect(self._on_export)
        btn_layout.addWidget(self.btn_export)

        layout.addLayout(btn_layout)

        # 填充人员卡片
        self._populate_person_cards()

        return panel

    def _make_button(self, text, color, border_color):
        btn = QPushButton(text)
        btn.setFixedHeight(34)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR_BG_CARD};
                color: {color};
                border: 1px solid {border_color};
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background: {COLOR_BG_CARD_HOVER};
                border-color: {COLOR_ACCENT};
                color: {COLOR_ACCENT};
            }}
        """)
        return btn

    def _create_left_header(self):
        header = QFrame()
        header.setFixedHeight(52)
        header.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLOR_BG_PANEL}, stop:1 rgba(0,212,170,0.08));
            border: none;
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
        """)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)

        title = QLabel("签到管理")
        title.setStyleSheet(f"""
            font-size: 16px; font-weight: 700; color: {COLOR_TEXT_PRIMARY};
            background: transparent; border: none;
        """)
        hl.addWidget(title)
        hl.addStretch()

        self.clock_label = QLabel()
        self.clock_label.setStyleSheet(f"""
            font-size: 12px; color: {COLOR_TEXT_SECONDARY};
            background: transparent; border: none;
        """)
        hl.addWidget(self.clock_label)

        # 时钟更新
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()

        return header

    def _update_clock(self):
        self.clock_label.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    def _populate_person_cards(self):
        """根据特征库填充人员卡片"""
        self.cards = {}
        records = self.sign_manager.records
        for display_name in records:
            rec = records[display_name]
            card = PersonCard(display_name, rec["person_id"])
            # 插入到 stretch 之前
            self.person_list_layout.insertWidget(
                self.person_list_layout.count() - 1, card
            )
            self.cards[display_name] = card

        # 初始统计
        self._refresh_stats()

    def _refresh_stats(self):
        total = self.sign_manager.total_count()
        signed = self.sign_manager.signed_count()
        remaining = self.sign_manager.remaining_count()
        self.stats_panel.update_stats(total, signed, remaining)

    def _on_reset(self):
        self.sign_manager.reset_all()
        for card in self.cards.values():
            card.set_signed(False)
        self._refresh_stats()
        # 重置线程侧签到状态，否则 _sign_emitted 未清会导致不再触发
        if self.recognition_thread is not None:
            self.recognition_thread.reset_sign_state()
        self._flash_button(self.btn_reset)

    def _on_export(self):
        """导出签到记录"""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getSaveFileName(
            self, "导出签到记录",
            f"signin_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV 文件 (*.csv)"
        )
        if not path:
            return
        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["姓名", "ID", "签到状态", "签到时间", "置信度"])
                for display_name, rec in self.sign_manager.records.items():
                    writer.writerow([
                        display_name, rec["person_id"],
                        "已签到" if rec["signed"] else "未签到",
                        rec["sign_time"] or "",
                        f"{rec['confidence']:.2f}" if rec["signed"] else ""
                    ])
            QMessageBox.information(self, "导出成功", f"记录已导出到:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))

    def _flash_button(self, btn):
        """按钮短暂闪烁反馈"""
        original = btn.styleSheet()
        btn.setStyleSheet(original.replace(
            "border: 1px solid", "border: 2px solid"
        ).replace(COLOR_BORDER, COLOR_ACCENT))
        QTimer.singleShot(300, lambda: btn.setStyleSheet(original))

    # ── 右侧面板 ─────────────────────────────────────────────
    def _create_right_panel(self):
        panel = QFrame()
        panel.setObjectName("rightPanel")
        panel.setStyleSheet(f"""
            #rightPanel {{
                background: {COLOR_BG_PANEL};
                border-radius: 12px;
                border: 1px solid {COLOR_BORDER};
            }}
        """)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 视频区域标题
        video_header = QFrame()
        video_header.setFixedHeight(44)
        video_header.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLOR_BG_PANEL}, stop:1 rgba(0,212,170,0.06));
            border: none;
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
        """)
        vh_layout = QHBoxLayout(video_header)
        vh_layout.setContentsMargins(16, 0, 16, 0)

        vid_title = QLabel("实时识别画面")
        vid_title.setStyleSheet(f"""
            font-size: 15px; font-weight: 700; color: {COLOR_TEXT_PRIMARY};
            background: transparent; border: none;
        """)
        vh_layout.addWidget(vid_title)
        vh_layout.addStretch()

        # 状态指示灯
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"""
            font-size: 10px; color: {COLOR_SUCCESS};
            background: transparent; border: none;
        """)
        vh_layout.addWidget(self.status_dot)

        self.status_text = QLabel(" 运行中")
        self.status_text.setStyleSheet(f"""
            font-size: 11px; color: {COLOR_TEXT_SECONDARY};
            background: transparent; border: none;
        """)
        vh_layout.addWidget(self.status_text)

        layout.addWidget(video_header)

        # 分割线
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {COLOR_DIVIDER}; border: none;")
        layout.addWidget(div)

        # 视频显示区
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(f"""
            background: {COLOR_BG_DARK};
            border: none;
            border-radius: 0px;
        """)
        # 占位文字
        self.video_label.setText("正在启动摄像头...")
        self.video_label.setStyleSheet(f"""
            background: {COLOR_BG_DARK};
            border: none;
            color: {COLOR_TEXT_SECONDARY};
            font-size: 16px;
        """)
        layout.addWidget(self.video_label, 1)

        # 分割线
        div2 = QFrame()
        div2.setFixedHeight(1)
        div2.setStyleSheet(f"background: {COLOR_DIVIDER}; border: none;")
        layout.addWidget(div2)

        # 底部控制栏
        control_bar = self._create_control_bar()
        layout.addWidget(control_bar)

        return panel

    def _create_control_bar(self):
        bar = QFrame()
        bar.setFixedHeight(56)
        bar.setStyleSheet(f"background: transparent; border: none;")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 4, 16, 8)
        layout.setSpacing(12)

        # FPS 显示
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setFixedWidth(70)
        self.fps_label.setStyleSheet(f"""
            font-size: 11px; font-weight: 600; color: {COLOR_TEXT_SECONDARY};
            background: transparent; border: none;
        """)
        layout.addWidget(self.fps_label)

        layout.addStretch()

        # 阈值标签
        thresh_label = QLabel("识别阈值")
        thresh_label.setStyleSheet(f"""
            font-size: 11px; color: {COLOR_TEXT_SECONDARY};
            background: transparent; border: none;
        """)
        layout.addWidget(thresh_label)

        # 阈值滑块
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setFixedWidth(160)
        self.threshold_slider.setRange(5, 60)
        self.threshold_slider.setValue(int(self.threshold * 100))
        self.threshold_slider.setSingleStep(5)
        self.threshold_slider.valueChanged.connect(self._on_threshold_change)
        layout.addWidget(self.threshold_slider)

        self.threshold_value_label = QLabel(f"{self.threshold:.2f}")
        self.threshold_value_label.setFixedWidth(36)
        self.threshold_value_label.setStyleSheet(f"""
            font-size: 13px; font-weight: 700; color: {COLOR_ACCENT};
            background: transparent; border: none;
        """)
        layout.addWidget(self.threshold_value_label)

        layout.addStretch()

        # 停止/启动按钮
        self.btn_toggle = QPushButton("停止识别")
        self.btn_toggle.setFixedHeight(34)
        self.btn_toggle.setFixedWidth(100)
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.clicked.connect(self._on_toggle)
        self.btn_toggle.setStyleSheet(f"""
            QPushButton {{
                background: rgba(255,82,82,0.15);
                color: {COLOR_DANGER};
                border: 1px solid rgba(255,82,82,0.3);
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: rgba(255,82,82,0.25);
            }}
        """)
        layout.addWidget(self.btn_toggle)

        # 截图按钮
        self.btn_screenshot = QPushButton("截图")
        self.btn_screenshot.setFixedHeight(34)
        self.btn_screenshot.setFixedWidth(60)
        self.btn_screenshot.setCursor(Qt.PointingHandCursor)
        self.btn_screenshot.clicked.connect(self._on_screenshot)
        self.btn_screenshot.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR_BG_CARD};
                color: {COLOR_TEXT_SECONDARY};
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLOR_BG_CARD_HOVER};
                color: {COLOR_ACCENT};
                border-color: {COLOR_ACCENT};
            }}
        """)
        layout.addWidget(self.btn_screenshot)

        return bar

    # ── 线程初始化 ───────────────────────────────────────────
    def _init_thread(self):
        self._running = True
        self._save_count = 0
        self._save_dir = os.path.join(project_root, "temp_results")
        os.makedirs(self._save_dir, exist_ok=True)

        self.recognition_thread = RecognitionThread(
            recognizer=self.recognizer,
            detector=self.detector,
            gallery_features=self.gallery_features,
            gallery_names=self.gallery_names,
            person_ids=self.gallery_person_ids,
            threshold=self.threshold,
            stabilize_frames=30,  # 连续10帧（~0.3s）匹配同一人才触发签到
        )
        self.recognition_thread.frame_ready.connect(self._on_frame_ready)
        self.recognition_thread.sign_in_event.connect(self._on_sign_in)
        self.recognition_thread.fps_update.connect(self._on_fps_update)
        self.recognition_thread.error_occurred.connect(self._on_error)
        self.recognition_thread.start()

    # ── 回调 ─────────────────────────────────────────────────
    def _on_frame_ready(self, display_frame, faces_info):
        """接收识别线程处理好的帧，更新显示"""
        try:
            h, w, ch = display_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_BGR888)

            # 缩放到标签大小，保持比例
            label_size = self.video_label.size()
            if label_size.width() > 10 and label_size.height() > 10:
                scaled = qimg.scaled(
                    label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            else:
                scaled = qimg.scaled(
                    800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )

            self.video_label.setPixmap(QPixmap.fromImage(scaled))
        except Exception:
            pass

    def _on_sign_in(self, display_name, confidence):
        """处理签到事件"""
        is_new = self.sign_manager.mark_signed(display_name, confidence)
        if is_new and display_name in self.cards:
            self.cards[display_name].set_signed(True, confidence=confidence)
            self._refresh_stats()
            # 闪烁动画
            self._flash_card(self.cards[display_name])

    def _on_fps_update(self, fps):
        color = COLOR_SUCCESS if fps > 20 else (COLOR_WARNING if fps > 10 else COLOR_DANGER)
        self.fps_label.setText(f"FPS: {fps:.0f}")
        self.fps_label.setStyleSheet(f"""
            font-size: 11px; font-weight: 600; color: {color};
            background: transparent; border: none;
        """)

    def _on_error(self, msg):
        self.status_dot.setStyleSheet(f"""
            font-size: 10px; color: {COLOR_DANGER};
            background: transparent; border: none;
        """)
        self.status_text.setText(f" 错误: {msg[:30]}")

    def _flash_card(self, card):
        """卡片签到高亮闪烁"""
        original = card.styleSheet()
        card.setStyleSheet(card.styleSheet().replace(
            "background: rgba(0, 212, 170, 0.12)",
            "background: rgba(0, 212, 170, 0.30)"
        ))
        QTimer.singleShot(800, lambda: card.setStyleSheet(original))

    # ── 控制回调 ─────────────────────────────────────────────
    def _on_threshold_change(self, val):
        t = val / 100.0
        self.threshold = t
        self.threshold_value_label.setText(f"{t:.2f}")
        if self.recognition_thread:
            self.recognition_thread.set_threshold(t)

    def _on_toggle(self):
        if self._running:
            self._running = False
            self.btn_toggle.setText("开始识别")
            self.btn_toggle.setStyleSheet(f"""
                QPushButton {{
                    background: rgba(0,212,170,0.15);
                    color: {COLOR_ACCENT};
                    border: 1px solid rgba(0,212,170,0.3);
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background: rgba(0,212,170,0.25);
                }}
            """)
            self.status_dot.setStyleSheet(f"""
                font-size: 10px; color: {COLOR_WARNING};
                background: transparent; border: none;
            """)
            self.status_text.setText(" 已暂停")
            # 停止线程
            if self.recognition_thread:
                self.recognition_thread.stop()
                self.recognition_thread = None
        else:
            self._running = True
            self.btn_toggle.setText("停止识别")
            self.btn_toggle.setStyleSheet(f"""
                QPushButton {{
                    background: rgba(255,82,82,0.15);
                    color: {COLOR_DANGER};
                    border: 1px solid rgba(255,82,82,0.3);
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background: rgba(255,82,82,0.25);
                }}
            """)
            self.status_dot.setStyleSheet(f"""
                font-size: 10px; color: {COLOR_SUCCESS};
                background: transparent; border: none;
            """)
            self.status_text.setText(" 运行中")
            # 重新创建线程
            self._init_thread()

    def _on_screenshot(self):
        """保存当前帧截图"""
        pixmap = self.video_label.pixmap()
        if pixmap is None:
            return
        self._save_count += 1
        path = os.path.join(self._save_dir, f"signin_{self._save_count:03d}.jpg")
        pixmap.save(path, "JPEG", 90)
        self._flash_button(self.btn_screenshot)

    # ── 关闭事件 ─────────────────────────────────────────────
    def closeEvent(self, event):
        if self.recognition_thread:
            self.recognition_thread.stop()
        event.accept()


# ─── 启动入口 ─────────────────────────────────────────────────
def main():
    # 高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("FaceSign")

    gallery_path = os.path.join(project_root, "galleries", "gallery_features_realtime.npz")
    model_path = os.path.join(project_root, "models", "adaface_ir50_ms1mv2.ckpt")

    window = SignInWindow(
        gallery_path=gallery_path,
        model_path=model_path,
        threshold=0.30,
    )
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

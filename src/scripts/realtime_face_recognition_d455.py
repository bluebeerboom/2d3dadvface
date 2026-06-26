"""
基于 Intel RealSense D455 的实时人脸识别系统
使用 pyrealsense2 读取 D455 彩色流，InsightFace 进行人脸检测，AdaFace 进行人脸识别
"""
import os
import sys
import json
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 先导入 cv2，再导入 insightface（避免版本冲突）
import cv2
import numpy as np
import pyrealsense2 as rs
import insightface
from insightface.app import FaceAnalysis

from src.utils.adaface_recognizer import AdaFaceRecognizer


class RealTimeFaceRecognition:
    """实时人脸识别系统"""

    def __init__(self, gallery_path='galleries/gallery_features_realtime.npz',
                 model_path='models/adaface_ir50_ms1mv2.ckpt',
                 recognition_threshold=0.20):
        """
        初始化实时识别系统

        Args:
            gallery_path: 特征库路径
            model_path: AdaFace 模型路径
            recognition_threshold: 识别阈值
        """
        print("=" * 70)
        print("实时人脸识别系统启动")
        print("=" * 70)

        # 加载特征库
        print("\n[1] 加载特征库...")
        if not os.path.exists(gallery_path):
            raise FileNotFoundError(f"特征库不存在: {gallery_path}")

        self.gallery_data = np.load(gallery_path, allow_pickle=True)
        self.gallery_features = self.gallery_data['features']
        self.gallery_names = self.gallery_data['names']
        self.gallery_person_ids = self.gallery_data['person_ids']

        print(f"  ✓ 加载 {len(self.gallery_names)} 个人")
        for i, name in enumerate(self.gallery_names):
            print(f"    {i + 1}. {name} (ID: {self.gallery_person_ids[i]})")

        # 加载元数据
        metadata_path = gallery_path.replace('.npz', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"  ✓ 元数据加载完成")

        # 初始化 AdaFace 识别器
        print("\n[2] 加载 AdaFace 模型...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.recognizer = AdaFaceRecognizer(
            model_path=model_path,
            device='cuda'
        )

        # 初始化 InsightFace 检测器
        print("\n[3] 加载 InsightFace 检测器...")
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        print(f"  可用的执行提供者: {available_providers}")

        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("  ✓ 使用 GPU (CUDA) 加速")
        else:
            providers = ['CPUExecutionProvider']
            print("  ⚠ 使用 CPU 运行")

        self.detector = FaceAnalysis(providers=providers)
        self.detector.prepare(ctx_id=0, det_size=(640, 640))
        print("  ✓ 检测器加载完成")

        # 识别阈值
        self.threshold = recognition_threshold

        # 统计信息
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        print("\n[4] 系统准备就绪")
        print("-" * 70)

    def open_camera(self):
        """打开 Intel RealSense D455 摄像头，只启用彩色流"""
        print("\n[5] 打开 Intel RealSense D455 摄像头...")

        self.pipeline = None
        self.config = None
        self.profile = None

        try:
            # 创建 RealSense 管线
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # 启用 D455 彩色流。这里使用 bgr8，后续 OpenCV/InsightFace 可以直接使用。
            # 如果 1280x720 不稳定，可以先用 640x480。
            self.color_width = 640
            self.color_height = 480
            self.color_fps = 30

            self.config.enable_stream(
                rs.stream.color,
                self.color_width,
                self.color_height,
                rs.format.bgr8,
                self.color_fps
            )

            # 启动摄像头
            self.profile = self.pipeline.start(self.config)

            # 预热摄像头，避免刚启动时读到空帧
            print("  正在预热 D455 摄像头...")
            for _ in range(30):
                self.pipeline.wait_for_frames(timeout_ms=5000)

            print("  ✓ D455 彩色摄像头启动成功")
            print(f"    分辨率: {self.color_width} x {self.color_height}")
            print(f"    FPS: {self.color_fps}")
            return True

        except Exception as e:
            print(f"❌ D455 摄像头打开失败: {e}")
            print("\n可能的解决方案:")
            print("  1. 确认 D455 已连接到 USB 3.0 接口")
            print("  2. 关闭 Intel RealSense Viewer、浏览器、相机等占用设备的软件")
            print("  3. 用 Intel RealSense Viewer 检查彩色流是否能正常打开")
            print("  4. 重新插拔 D455 或重启电脑")
            try:
                if self.pipeline is not None:
                    self.pipeline.stop()
            except Exception:
                pass
            return False

    def read_d455_frame(self):
        """读取 Intel RealSense D455 彩色帧，返回 BGR 图像"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("❌ 未获取到 D455 彩色帧")
                return False, None

            frame = np.asanyarray(color_frame.get_data())

            if frame is None or frame.size == 0:
                print("❌ D455 彩色帧为空")
                return False, None

            return True, frame

        except Exception as e:
            print(f"❌ D455 读取帧失败: {e}")
            return False, None

    def close_camera(self):
        """关闭 Intel RealSense D455 摄像头"""
        try:
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                self.pipeline.stop()
                print("✓ D455 摄像头已关闭")
        except Exception as e:
            print(f"⚠ 关闭 D455 摄像头时出现异常: {e}")

    def process_frame(self, frame):
        """
        处理单帧图像

        Args:
            frame: numpy array (BGR)

        Returns:
            display_frame: 带标注的图像
            faces_info: 人脸信息列表
        """
        display_frame = frame.copy()
        faces_info = []

        # 检测人脸
        try:
            faces = self.detector.get(frame)
        except Exception as e:
            print(f"检测错误: {e}")
            faces = []

        # 处理每个检测到的人脸
        for i, face in enumerate(faces):
            # 获取边界框
            bbox = face.bbox.astype(np.int32)

            # 使用 InsightFace 检测到的边界框裁剪人脸（添加边距）
            x1, y1, x2, y2 = bbox
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)

            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            # 使用 AdaFace 提取特征并识别
            try:
                query_feature = self.recognizer.extract_feature(face_img)

                # 识别
                result = self.recognizer.recognize(
                    query_feature=query_feature,
                    gallery_features=self.gallery_features,
                    gallery_names=self.gallery_names,
                    threshold=self.threshold
                )

                faces_info.append({
                    'bbox': bbox,
                    'name': result['name'],
                    'similarity': result['similarity'],
                    'matched': result['matched']
                })

            except Exception as e:
                print(f"识别错误: {e}")
                faces_info.append({
                    'bbox': bbox,
                    'name': 'Error',
                    'similarity': 0.0,
                    'matched': False
                })

        # 绘制结果
        for info in faces_info:
            bbox = info['bbox']
            name = info['name']
            similarity = info['similarity']
            matched = info['matched']

            # 根据匹配结果选择颜色
            if matched:
                color = (0, 255, 0)  # 绿色 - 匹配
            else:
                color = (0, 0, 255)  # 红色 - 未匹配

            # 绘制边界框
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # 准备标签
            label = f"{name}"
            sub_label = f"{similarity:.2f}"

            # 绘制主标签背景
            font_scale = 0.6
            thickness = 2
            (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(display_frame,
                          (bbox[0], bbox[1] - h1 - 10),
                          (bbox[0] + w1 + 10, bbox[1]),
                          color, -1)

            # 绘制主标签
            cv2.putText(display_frame, label,
                        (bbox[0] + 5, bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # 绘制相似度
            font_scale_small = 0.4
            (w2, h2), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, 1)
            cv2.rectangle(display_frame,
                          (bbox[2] - w2 - 5, bbox[1]),
                          (bbox[2] + 5, bbox[1] + h2 + 5),
                          (0, 0, 0), -1)
            cv2.putText(display_frame, sub_label,
                        (bbox[2] - w2, bbox[1] + h2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_small,
                        (0, 255, 0) if matched else (0, 0, 255), 1)

        # 显示 FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.fps_start_time
        if elapsed_time > 1.0:
            self.current_fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.fps_start_time = time.time()

        fps_color = (0, 255, 0) if self.current_fps > 20 else (0, 255, 255) if self.current_fps > 10 else (0, 0, 255)
        cv2.putText(display_frame, f"FPS: {self.current_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

        # 显示检测到的人脸数量
        cv2.putText(display_frame, f"Faces: {len(faces_info)}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 显示阈值信息
        cv2.putText(display_frame, f"Threshold: {self.threshold:.2f}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 在右下角显示操作提示
        hints = [
            "Q: Quit",
            "T: Threshold+",
            "G: Threshold-",
            "S: Save"
        ]

        hint_y = display_frame.shape[0] - 20
        for i, hint in enumerate(hints):
            cv2.putText(display_frame, hint,
                        (display_frame.shape[1] - 150, hint_y - i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return display_frame, faces_info

    def run(self):
        """运行实时识别"""
        if not self.open_camera():
            return

        print("\n[6] 开始实时识别...")
        print("-" * 70)
        print("快捷键说明:")
        print("  Q / ESC - 退出")
        print("  T / t   - 提高阈值 (+0.05)")
        print("  G / g   - 降低阈值 (-0.05)")
        print("  S       - 保存截图")
        print("-" * 70)

        save_count = 0
        save_dir = os.path.join(project_root, 'temp_results')
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n截图保存目录: {save_dir}")

        try:
            while True:
                ret, frame = self.read_d455_frame()

                if not ret or frame is None:
                    print("❌ 无法读取 D455 彩色画面，跳过当前帧")
                    time.sleep(0.03)
                    continue

                # 处理帧
                display_frame, faces_info = self.process_frame(frame)

                # 显示
                cv2.imshow('Real-time Face Recognition - D455', display_frame)

                # 按键处理
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q') or key == 27:
                    print("\n退出程序...")
                    break
                elif key == ord('t') or key == ord('T'):
                    self.threshold = min(1.0, self.threshold + 0.05)
                    print(f"✓ 阈值提高到: {self.threshold:.2f}")
                elif key == ord('g') or key == ord('G'):
                    self.threshold = max(0.0, self.threshold - 0.05)
                    print(f"✓ 阈值降低到: {self.threshold:.2f}")
                elif key == ord('s') or key == ord('S'):
                    save_count += 1
                    save_path = os.path.join(save_dir, f"recognition_{save_count:03d}.jpg")

                    # 使用 imencode 解决中文路径问题
                    try:
                        is_success, buffer = cv2.imencode(".jpg", display_frame)
                        if is_success:
                            with open(save_path, 'wb') as f:
                                f.write(buffer.tobytes())
                            print(f"✓ 截图已保存: {save_path}")
                        else:
                            print(f"❌ 截图保存失败: 图像编码错误")
                    except Exception as e:
                        print(f"❌ 截图保存失败: {e}")

        finally:
            # 清理资源
            self.close_camera()
            cv2.destroyAllWindows()

            print("\n" + "=" * 70)
            print("程序已结束")
            print("=" * 70)
            print(f"总共保存截图: {save_count} 张")


if __name__ == '__main__':
    # 配置路径
    gallery_path = os.path.join(project_root, 'galleries', 'gallery_features_realtime.npz')
    model_path = os.path.join(project_root, 'models', 'adaface_ir50_ms1mv2.ckpt')

    # 检查文件是否存在
    if not os.path.exists(gallery_path):
        print(f"❌ 特征库不存在: {gallery_path}")
        print("请先运行 extract_gallery_features_realtime.py 提取特征")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        sys.exit(1)

    # 创建并运行系统
    system = RealTimeFaceRecognition(
        gallery_path=gallery_path,
        model_path=model_path,
        recognition_threshold=0.3
    )

    system.run()

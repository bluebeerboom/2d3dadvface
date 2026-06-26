"""
实时摄像头人脸识别系统
使用 InsightFace 进行人脸检测，AdaFace 进行人脸识别
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
        print("="*70)
        print("实时人脸识别系统启动")
        print("="*70)
        
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
            print(f"    {i+1}. {name} (ID: {self.gallery_person_ids[i]})")
        
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
        """打开摄像头"""
        print("\n[5] 打开摄像头...")
        
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_ANY, "Any"),
        ]
        
        self.cap = None
        for backend_code, backend_name in backends:
            print(f"  尝试 {backend_name}...")
            self.cap = cv2.VideoCapture(0, backend_code)
            
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print(f"  ✓ 使用 {backend_name} 成功")
                    break
                else:
                    self.cap.release()
                    self.cap = None
        
        if self.cap is None or not self.cap.isOpened():
            print("❌ 无法打开摄像头")
            print("\n可能的解决方案:")
            print("  1. 检查摄像头是否被其他程序占用")
            print("  2. 检查摄像头权限设置")
            print("  3. 尝试重启摄像头或计算机")
            return False
        
        # 设置参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"  ✓ 摄像头已打开")
        print(f"    分辨率: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        return True
    
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
                         (bbox[0], bbox[1]-h1-10), 
                         (bbox[0]+w1+10, bbox[1]), 
                         color, -1)
            
            # 绘制主标签
            cv2.putText(display_frame, label, 
                       (bbox[0]+5, bbox[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # 绘制相似度
            font_scale_small = 0.4
            (w2, h2), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, 1)
            cv2.rectangle(display_frame, 
                         (bbox[2]-w2-5, bbox[1]), 
                         (bbox[2]+5, bbox[1]+h2+5), 
                         (0, 0, 0), -1)
            cv2.putText(display_frame, sub_label, 
                       (bbox[2]-w2, bbox[1]+h2),
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
                       (display_frame.shape[1] - 150, hint_y - i*25),
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
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    print("\n尝试重新初始化摄像头...")
                    self.cap.release()
                    time.sleep(1)
                    
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(0)
                    
                    if self.cap.isOpened():
                        print("✓ 摄像头重新打开成功")
                        continue
                    else:
                        print("❌ 无法重新打开摄像头，退出程序")
                        break
                
                # 处理帧
                display_frame, faces_info = self.process_frame(frame)
                
                # 显示
                cv2.imshow('Real-time Face Recognition', display_frame)
                
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
            self.cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*70)
            print("程序已结束")
            print("="*70)
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
        recognition_threshold=0.4
    )
    
    system.run()

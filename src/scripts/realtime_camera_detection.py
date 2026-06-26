"""
实时摄像头人脸检测系统
使用InsightFace进行实时人脸检测、识别和属性分析
"""
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
import sys
import platform
import os

print("="*70)
print("InsightFace 实时摄像头人脸检测系统")
print("="*70)
print(f"\n系统信息:")
print(f"  操作系统: {platform.system()} {platform.release()}")
print(f"  Python版本: {sys.version.split()[0]}")
print(f"  OpenCV版本: {cv2.__version__}")
print(f"  可用摄像头后端: DirectShow, MSMF, Any")

# 初始化模型
print("\n[1] 加载模型...")
# 检查可用的执行提供者
import onnxruntime as ort
available_providers = ort.get_available_providers()
print(f"  可用的执行提供者: {available_providers}")

# 优先使用GPU，如果不可用则回退到CPU
if 'CUDAExecutionProvider' in available_providers:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("  ✓ 使用 GPU (CUDA) 加速")
else:
    providers = ['CPUExecutionProvider']
    print("  ⚠ 使用 CPU 运行（未检测到GPU支持）")

app = FaceAnalysis(providers=providers)
app.prepare(ctx_id=0, det_size=(640, 640))
print("✓ 模型加载完成")

# 打开摄像头
print("\n[2] 打开摄像头...")

# 尝试不同的后端
backends = [
    (cv2.CAP_DSHOW, "DirectShow"),  # Windows DirectShow (更稳定)
    (cv2.CAP_MSMF, "MSMF"),         # Microsoft Media Foundation
    (cv2.CAP_ANY, "Any"),           # 自动选择
]

cap = None
for backend_code, backend_name in backends:
    print(f"  尝试使用 {backend_name} 后端...")
    cap = cv2.VideoCapture(0, backend_code)
    
    if cap.isOpened():
        # 测试是否能读取帧
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print(f"✓ 使用 {backend_name} 后端成功")
            break
        else:
            print(f"  ⚠ {backend_name} 后端无法读取帧，尝试下一个...")
            cap.release()
            cap = None
    else:
        print(f"  ⚠ {backend_name} 后端无法打开")

if cap is None or not cap.isOpened():
    print("❌ 无法打开摄像头")
    print("\n可能的解决方案:")
    print("  1. 检查摄像头是否被其他程序占用")
    print("  2. 检查摄像头权限设置")
    print("  3. 尝试重启摄像头或计算机")
    exit()

# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print(f"✓ 摄像头已打开")
print(f"  分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"  FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")

# 统计信息
frame_count = 0
fps_start_time = time.time()
current_fps = 0

print("\n[3] 开始实时检测...")
print("-" * 70)
print("快捷键说明:")
print("  Q / ESC - 退出程序")
print("  S       - 保存当前帧截图")
print("  R       - 重置统计信息")
print("  F       - 切换全屏模式")
print("-" * 70)

fullscreen = False
save_count = 0

# 创建保存目录
# 从 src/scripts 向上一级到 src，再向上一级到项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
save_dir = os.path.join(project_root, 'temp_results')
os.makedirs(save_dir, exist_ok=True)
print(f"\n截图保存目录: {save_dir}")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ 无法读取摄像头画面")
        print("\n尝试重新初始化摄像头...")
        cap.release()
        time.sleep(1)
        
        # 重新打开摄像头
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✓ 摄像头重新打开成功")
            continue
        else:
            print("❌ 无法重新打开摄像头，退出程序")
            break
    
    frame_count += 1
    
    # 计算FPS
    elapsed_time = time.time() - fps_start_time
    if elapsed_time > 1.0:
        current_fps = frame_count / elapsed_time
        frame_count = 0
        fps_start_time = time.time()
    
    # 复制帧用于显示
    display_frame = frame.copy()
    
    # 检测人脸
    try:
        faces = app.get(frame)
    except Exception as e:
        print(f"检测错误: {e}")
        faces = []
    
    # 绘制检测结果
    for i, face in enumerate(faces):
        # 获取边界框
        bbox = face.bbox.astype(np.int32)
        
        # 根据性别选择颜色（蓝色=男，粉色=女）
        color = (255, 0, 0) if face.gender == 0 else (255, 0, 255)
        
        # 绘制边界框
        cv2.rectangle(display_frame, 
                     (bbox[0], bbox[1]), 
                     (bbox[2], bbox[3]), 
                     color, 2)
        
        # 绘制关键点（眼睛、鼻子、嘴巴）
        if hasattr(face, 'kps') and face.kps is not None:
            for kpt in face.kps:
                cv2.circle(display_frame, (int(kpt[0]), int(kpt[1])), 2, (0, 255, 0), -1)
        
        # 准备标签信息
        gender_text = "M" if face.gender == 0 else "F"
        age_text = f"{face.age}y"
        
        # 置信度
        det_score = face.det_score if hasattr(face, 'det_score') else 0
        
        # 主标签：性别 + 年龄
        main_label = f"#{i+1} {gender_text} {age_text}"
        
        # 副标签：置信度
        sub_label = f"{det_score:.2f}"
        
        # 绘制主标签背景
        font_scale = 0.6
        thickness = 2
        (w1, h1), _ = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(display_frame, 
                     (bbox[0], bbox[1]-h1-10), 
                     (bbox[0]+w1+10, bbox[1]), 
                     color, -1)
        
        # 绘制主标签文字
        cv2.putText(display_frame, main_label, 
                   (bbox[0]+5, bbox[1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # 绘制副标签（置信度）
        font_scale_small = 0.4
        (w2, h2), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, 1)
        cv2.rectangle(display_frame, 
                     (bbox[2]-w2-5, bbox[1]), 
                     (bbox[2]+5, bbox[1]+h2+5), 
                     (0, 0, 0), -1)
        cv2.putText(display_frame, sub_label, 
                   (bbox[2]-w2, bbox[1]+h2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0, 255, 0), 1)
    
    # 在左上角显示FPS和人脸数量
    info_y = 30
    
    # FPS
    fps_color = (0, 255, 0) if current_fps > 20 else (0, 255, 255) if current_fps > 10 else (0, 0, 255)
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", 
               (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
    
    # 人脸数量
    cv2.putText(display_frame, f"Faces: {len(faces)}", 
               (10, info_y + 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 分辨率信息
    h, w = frame.shape[:2]
    cv2.putText(display_frame, f"Resolution: {w}x{h}", 
               (10, info_y + 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # 在右下角显示操作提示
    hints = [
        "Q: Quit",
        "S: Save",
        "R: Reset",
        "F: Fullscreen"
    ]
    
    hint_y = display_frame.shape[0] - 20
    for i, hint in enumerate(hints):
        cv2.putText(display_frame, hint, 
                   (display_frame.shape[1] - 120, hint_y - i*25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # 显示画面
    if fullscreen:
        cv2.namedWindow('Real-time Face Detection', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Real-time Face Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    cv2.imshow('Real-time Face Detection', display_frame)
    
    # 按键处理
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q') or key == 27:  # Q或ESC退出
        print("\n退出程序...")
        break
    elif key == ord('s') or key == ord('S'):  # S保存截图
        save_count += 1
        save_path = os.path.join(save_dir, f"screenshot_{save_count:03d}.jpg")
        
        # 使用imencode解决中文路径问题
        try:
            # 将图像编码为JPEG格式
            is_success, buffer = cv2.imencode(".jpg", display_frame)
            if is_success:
                # 将buffer写入文件
                with open(save_path, 'wb') as f:
                    f.write(buffer.tobytes())
                print(f"✓ 截图已保存: {save_path}")
            else:
                print(f"❌ 截图保存失败: 图像编码错误")
        except Exception as e:
            print(f"❌ 截图保存失败: {e}")
    elif key == ord('r') or key == ord('R'):  # R重置统计
        frame_count = 0
        fps_start_time = time.time()
        save_count = 0
        print("✓ 统计信息已重置")
    elif key == ord('f') or key == ord('F'):  # F切换全屏
        fullscreen = not fullscreen
        print(f"✓ 全屏模式: {'开启' if fullscreen else '关闭'}")

# 清理资源
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("程序已结束")
print("="*70)
print(f"总共保存截图: {save_count} 张")

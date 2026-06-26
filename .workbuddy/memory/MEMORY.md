# 项目记忆 (2D3DAdvFace)

## 环境信息
- conda环境: `2d3dadvface` (Python 3.10, torch 2.4.1+cu121)
- 位置: `C:\Users\CY406\.conda\envs\2d3dadvface\`
- 激活方式: `conda activate 2d3dadvface`

## 项目结构
- `src/scripts/realtime_face_recognition_d455.py` - D455实时人脸识别 (pyrealsense2, 640x480@30fps, OpenCV GUI)
- `src/scripts/realtime_face_recognition.py` - 通用摄像头实时人脸识别
- `src/scripts/extract_gallery_features_realtime.py` - 特征库构建脚本
- `src/utils/adaface_recognizer.py` - AdaFace识别器
- `galleries/gallery_features_realtime.npz` - 12人特征库
- `models/adaface_ir50_ms1mv2.ckpt` - AdaFace模型 (IR-50 backbone, 700MB)
- insightface模型: `C:\Users\CY406\.insightface\models\buffalo_l\` (5个ONNX文件)

## 已知问题与解决方案
1. insightface 0.7.3 安装: 因缺少MSVC, Cython扩展无法编译。已手动安装并patch了mesh模块的导入(fallback到None)
2. pip 26.x有zipfile排序bug, 使用pip 25.0.1解决
3. conda run 不支持多行-c脚本, 需使用文件方式运行
4. opencv-python-headless 与 opencv-python 共存导致 cv2.imshow() 报 "-2:Unspecified error"。已卸载 opencv-python-headless, 保留 opencv-python 4.13.0.92 (2026-05-26)
5. 运行脚本需用 python -u 避免stdout缓冲导致无输出

## 用户习惯
- 中文交流, 风格简洁直接
- 研究方向: 物联网安全 + 深度强化学习 (IEEE TIFS论文)
- 工作环境: Windows工作站, 有NVIDIA GPU (CUDA 12.1)

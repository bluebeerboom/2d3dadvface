# AdaFace 实时人脸识别系统使用指南

## 📋 概述

本系统使用 AdaFace 预训练模型进行高精度人脸识别，支持从自定义数据库提取特征并进行实时摄像头识别。

## 🗂️ 文件结构

```
项目根目录/
├── Dataset/                          # 你的12人数据库
│   ├── 001/                         # 每人一个文件夹
│   │   ├── photo1.jpg
│   │   ├── photo2.jpg
│   │   └── ... (5张照片)
│   ├── 002/
│   └── ... (共12人)
│
├── galleries/                        # 特征库存储位置
│   └── gallery_features_realtime.npz  # 生成的特征库文件
│
├── models/
│   └── adaface_ir50_ms1mv2.ckpt     # AdaFace 预训练模型
│
├── src/
│   ├── utils/
│   │   └── adaface_recognizer.py    # ✨ AdaFace 推理工具类
│   └── scripts/
│       └── extract_gallery_features_realtime.py  # ✨ 特征提取脚本
│
└── AdaFace-master/                   # AdaFace 源码
```

## 🚀 使用步骤

### 第一步：准备数据库

确保 `Dataset/` 文件夹中包含12个人的照片，每人一个文件夹，每人至少5张照片：

```
Dataset/
├── 001/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   ├── photo3.jpg
│   ├── photo4.jpg
│   └── photo5.jpg
├── 002/
│   └── ...
└── 012/
    └── ...
```

### 第二步：提取特征

运行特征提取脚本，从12人的照片中提取 AdaFace 特征：

```bash
cd "d:\collegedata\2026\研二\警学项目申报\2d3dadvface"
python src/scripts/extract_gallery_features_realtime.py
```

**执行过程：**
1. 加载 AdaFace 预训练模型
2. 遍历 Dataset 中的12个人
3. 对每个人的5张照片提取特征
4. 计算每个人的平均特征（更鲁棒）
5. 保存到 `galleries/gallery_features_realtime.npz`

**输出文件：**
- `galleries/gallery_features_realtime.npz` - 特征库（包含12人的512维特征向量）
- `galleries/gallery_features_realtime_metadata.json` - 元数据（包含人员ID和名称）

### 第三步：运行实时识别

有两种方式启动实时识别系统：

**方式1：使用 PowerShell 脚本（推荐）**
```powershell
.\run_realtime_recognition.ps1
```

**方式2：直接运行 Python 脚本**
```bash
python src/scripts/realtime_face_recognition.py
```

**系统功能：**
1. ✅ 加载 gallery_features_realtime.npz 中的12人特征
2. ✅ 使用 InsightFace 检测摄像头中的人脸
3. ✅ 使用 AdaFace 提取检测到的人脸特征
4. ✅ 与特征库比对，识别出是谁
5. ✅ 在视频画面上显示姓名和相似度
6. ✅ 支持动态调整识别阈值（T/G键）
7. ✅ 支持保存截图（S键）

## 🔧 核心组件说明

### 1. AdaFaceRecognizer 工具类

**位置：** `src/utils/adaface_recognizer.py`

**主要功能：**
- 加载 AdaFace 预训练模型
- 预处理图像（BGR → RGB → 112x112 → 归一化）
- 提取512维人脸特征
- 计算余弦相似度
- 识别人脸（与特征库比对）

**使用方法：**
```python
from src.utils.adaface_recognizer import AdaFaceRecognizer

# 初始化
recognizer = AdaFaceRecognizer(
    model_path='models/adaface_ir50_ms1mv2.ckpt',
    device='cuda'  # 或 'cpu'
)

# 提取单张图像的特征
import cv2
img = cv2.imread('photo.jpg')
feature = recognizer.extract_feature(img)  # 返回512维向量

# 识别
result = recognizer.recognize(
    query_feature=query_feat,
    gallery_features=gallery_feats,
    gallery_names=gallery_names,
    threshold=0.4
)
# 返回: {'name': 'Person_001', 'similarity': 0.85, 'matched': True}
```

### 2. 特征提取脚本

**位置：** `src/scripts/extract_gallery_features_realtime.py`

**功能：**
- 从 Dataset 数据库读取12人的照片
- 批量提取特征
- 计算每人的平均特征
- 保存为 .npz 格式

**配置参数：**
```python
database_path = 'Dataset/'                    # 数据库路径
output_path = 'galleries/gallery_features_realtime.npz'  # 输出路径
model_path = 'models/adaface_ir50_ms1mv2.ckpt'           # 模型路径
```

## 📊 技术细节

### AdaFace 模型特点

- **架构：** ResNet-50 (IR-50)
- **输入：** 112x112 BGR 图像
- **输出：** 512维特征向量
- **训练数据：** MS1MV2 (约500万人)
- **优势：** 对低质量图像有更强的鲁棒性

### 特征处理流程

```
原始图像 (BGR) 
    ↓
转换为 RGB
    ↓
调整大小到 112x112
    ↓
归一化: (x/255 - 0.5) / 0.5
    ↓
AdaFace 模型提取特征
    ↓
L2 归一化
    ↓
512维特征向量
```

### 识别原理

1. **特征提取：** 对摄像头中的人脸提取512维特征
2. **相似度计算：** 计算与特征库中每个特征的余弦相似度
3. **匹配判断：** 
   - 找到最大相似度
   - 如果 ≥ 阈值（默认0.4），则识别为该人
   - 否则标记为 "Unknown"

## ⚙️ 环境要求

```bash
# Python 依赖
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.20.0
Pillow>=8.0.0
tqdm>=4.60.0

# GPU 加速（可选但推荐）
onnxruntime-gpu>=1.20.0
```

## 🎯 下一步工作

需要创建实时摄像头识别脚本 `src/scripts/realtime_face_recognition.py`，实现：

1. ✅ 加载 gallery_features_realtime.npz
2. ✅ 初始化 AdaFaceRecognizer
3. ✅ 初始化 InsightFace 检测器
4. ✅ 打开摄像头
5. ✅ 实时检测人脸
6. ✅ 提取特征并识别
7. ✅ 显示识别结果（姓名、相似度）
8. ✅ 支持快捷键调整阈值

## 📝 注意事项

1. **模型路径：** 确保 `models/adaface_ir50_ms1mv2.ckpt` 存在
2. **数据库结构：** Dataset 文件夹必须是 `数字ID/照片.jpg` 的结构
3. **GPU 支持：** 如果有 NVIDIA GPU，建议使用 CUDA 加速
4. **识别阈值：** 默认0.4，可根据实际情况调整（范围0-1）
   - 调高：更严格，减少误识别
   - 调低：更宽松，提高召回率

## 🔍 常见问题

**Q: 为什么提取的是平均特征而不是单张照片的特征？**
A: 多张照片的平均特征更鲁棒，能更好地代表该人的面部特征，减少单张照片质量不佳的影响。

**Q: 识别阈值应该设置为多少？**
A: 建议从0.4开始测试：
- 如果误识别太多，提高到0.5-0.6
- 如果漏识别太多，降低到0.3-0.35

**Q: 可以添加新的人吗？**
A: 可以，在 Dataset 中添加新文件夹和照片，然后重新运行特征提取脚本。

---

**创建时间：** 2026-05-09  
**版本：** 1.0

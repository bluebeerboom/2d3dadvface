# 实时人脸识别系统使用指南

## 📋 系统概述

本系统实现了基于 AdaFace 的实时人脸识别功能，能够：
- ✅ 从摄像头实时检测人脸
- ✅ 识别数据库中的12个已知人员
- ✅ 显示识别结果和相似度
- ✅ 支持动态调整识别阈值

## 🗂️ 文件结构

```
项目根目录/
├── Dataset/                              # 12人数据库（每人5张照片）
│   ├── 001/ ~ 012/
│
├── galleries/                            # 特征库
│   ├── gallery_features_realtime.npz     # ✨ 12人的特征向量
│   └── gallery_features_realtime_metadata.json
│
├── models/
│   └── adaface_ir50_ms1mv2.ckpt         # AdaFace 预训练模型
│
├── src/
│   ├── utils/
│   │   └── adaface_recognizer.py        # AdaFace 推理工具类
│   └── scripts/
│       ├── extract_gallery_features_realtime.py  # 特征提取脚本
│       └── realtime_face_recognition.py          # ✨ 实时识别脚本
│
└── AdaFace-master/                       # AdaFace 源码
```

## 🚀 快速开始

### 第一步：确认特征库已生成

确保已经运行过特征提取脚本，生成了 `galleries/gallery_features_realtime.npz`：

```bash
cd "d:\collegedata\2026\研二\警学项目申报\2d3dadvface"
python src/scripts/extract_gallery_features_realtime.py
```

**预期输出：**
```
✓ 成功提取 12 个人的特征
  特征维度: (12, 512)
  人员名单: Person_001, Person_002, ..., Person_012
```

### 第二步：运行实时识别系统

```bash
python src/scripts/realtime_face_recognition.py
```

**系统会自动：**
1. 加载特征库（12人）
2. 加载 AdaFace 模型
3. 加载 InsightFace 检测器
4. 打开摄像头
5. 开始实时识别

## 🎮 操作说明

### 快捷键

| 按键 | 功能 | 说明 |
|------|------|------|
| **Q / ESC** | 退出程序 | 关闭摄像头并退出 |
| **T / t** | 提高阈值 | 增加0.05，更严格 |
| **G / g** | 降低阈值 | 减少0.05，更宽松 |
| **S** | 保存截图 | 保存到 `temp_results/` 目录 |

### 识别阈值调整

**默认阈值：0.4**

- **调高阈值（按 T）**：
  - 更严格，减少误识别
  - 适合光线好、正脸清晰的场景
  - 建议范围：0.4 ~ 0.6

- **调低阈值（按 G）**：
  - 更宽松，提高召回率
  - 适合光线暗、侧脸、遮挡的场景
  - 建议范围：0.3 ~ 0.4

### 界面显示

```
┌─────────────────────────────────────┐
│ FPS: 25.3                           │  ← 左上角：帧率
│ Faces: 2                            │  ← 检测到的人脸数量
│ Threshold: 0.40                     │  ← 当前识别阈值
│                                     │
│   ┌──────────┐                      │
│   │ Person_001│ 0.85               │  ← 绿色框：匹配成功
│   └──────────┘                      │     显示姓名和相似度
│                                     │
│   ┌──────────┐                      │
│   │ Unknown  │ 0.32               │  ← 红色框：未匹配
│   └──────────┘                      │     相似度低于阈值
│                                     │
│                          Q: Quit    │  ← 右下角：操作提示
│                          T: Thresh+ │
│                          G: Thresh- │
│                          S: Save    │
└─────────────────────────────────────┘
```

## 🔧 技术细节

### 工作流程

```
摄像头画面
    ↓
InsightFace 检测人脸
    ↓
裁剪人脸区域（添加边距）
    ↓
AdaFace 提取512维特征
    ↓
与特征库比对（余弦相似度）
    ↓
找到最相似的人
    ↓
判断是否 ≥ 阈值
    ↓
显示结果（姓名 + 相似度）
```

### 识别原理

1. **人脸检测**：使用 InsightFace 检测画面中的人脸位置和关键点
2. **特征提取**：使用 AdaFace 模型提取512维特征向量
3. **相似度计算**：计算查询特征与特征库中每个特征的余弦相似度
4. **匹配判断**：
   - 找到最大相似度的人
   - 如果相似度 ≥ 阈值 → 识别为该人（绿色框）
   - 否则 → 标记为 "Unknown"（红色框）

### 性能优化

- **GPU 加速**：自动检测并使用 CUDA（如果可用）
- **批量处理**：同时处理多个人脸
- **缓存机制**：特征库一次性加载到内存

## 📊 识别效果优化

### 提高识别准确率

1. **数据库质量**：
   - 每人至少5张照片
   - 不同角度（正面、侧面）
   - 不同表情（微笑、正常）
   - 不同光照条件

2. **识别环境**：
   - 保证充足的光线
   - 尽量正对摄像头
   - 避免严重遮挡

3. **阈值调整**：
   - 初始值：0.4
   - 如果误识别多 → 提高到 0.5
   - 如果漏识别多 → 降低到 0.35

### 常见问题

**Q: 为什么有时候识别成 Unknown？**
A: 可能原因：
- 相似度低于阈值（可以尝试降低阈值）
- 该人不在数据库中
- 光线太暗或角度太差

**Q: 为什么会识别错误？**
A: 可能原因：
- 阈值太低，导致误匹配
- 两个人长得很像
- 建议提高阈值到 0.5 或更高

**Q: FPS 很低怎么办？**
A: 可能原因：
- 使用了 CPU 而非 GPU
- 分辨率太高
- 建议：降低摄像头分辨率或使用 GPU

## 🔍 调试技巧

### 查看识别详情

在代码中可以添加调试信息：

```python
# 在 process_frame 方法中添加
print(f"识别结果: {name}, 相似度: {similarity:.3f}, 匹配: {matched}")
```

### 测试单张图片

可以创建测试脚本验证识别效果：

```python
from src.utils.adaface_recognizer import AdaFaceRecognizer
import cv2
import numpy as np

# 加载模型和特征库
recognizer = AdaFaceRecognizer()
gallery_data = np.load('galleries/gallery_features_realtime.npz')

# 读取测试图片
img = cv2.imread('test.jpg')
feature = recognizer.extract_feature(img)

# 识别
result = recognizer.recognize(
    query_feature=feature,
    gallery_features=gallery_data['features'],
    gallery_names=gallery_data['names'],
    threshold=0.4
)

print(f"识别结果: {result['name']}")
print(f"相似度: {result['similarity']:.3f}")
```

## 📝 注意事项

1. **首次运行较慢**：模型加载需要几秒时间
2. **GPU 显存需求**：约 2-3GB
3. **中文路径支持**：已修复，可正常使用
4. **摄像头权限**：确保允许程序访问摄像头
5. **多人识别**：支持同时识别多个人脸

## 🎯 下一步扩展

可以进一步优化的方向：

1. **添加人员注册功能**：通过摄像头直接添加新人
2. **历史记录**：保存识别记录和时间戳
3. **Web 界面**：提供浏览器访问接口
4. **视频录制**：录制识别过程的视频
5. **报警功能**：识别到陌生人时发出警报

---

**创建时间：** 2026-05-09  
**版本：** 1.0  
**作者：** AI Assistant

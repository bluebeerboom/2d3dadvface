# Face Recognition System for Access Control
## 人脸识别签到系统

基于 AdaFace + InsightFace 的实时人脸识别签到系统，支持 Intel RealSense D455 摄像头。**v4.0 引入 5 点人脸对齐 + ArcFace/AdaFace 双通道融合，识别准确度大幅提升；v5.1 新增鲁班猫 BTB（RK3588）+ RKNNLite2 + PySide6 边缘部署版本。**

---

## 📋 目录

- [项目简介](#项目简介)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [实时签到系统](#实时签到系统)
- [数据集说明](#数据集说明)
- [使用方法](#使用方法)
- [技术细节](#技术细节)
- [RK3588 / 鲁班猫BTB边缘部署版本](#rk3588--鲁班猫btb边缘部署版本)
- [常见问题](#常见问题)
- [性能指标](#性能指标)

---

## 项目简介

本系统实现了完整的人脸识别流程：

1. **训练阶段**：使用 CFP 数据集训练 ArcFace 模型
2. **建库阶段**：从 CFP 数据集提取 500 人的特征向量建立特征库
3. **测试阶段**：在遮挡数据集（occluded）上测试识别准确率
4. **实时签到**：基于 D455 摄像头 + AdaFace 的实时人脸识别签到系统
5. **边缘部署**：基于鲁班猫 BTB（RK3588）+ RKNNLite2 + PySide6 的 NPU 加速签到系统

### 核心特点

✅ **v4.0 对齐管线**：5 点关键点相似变换对齐，类内相似度从 ~0.35 提升至 0.49-0.78  
✅ **双通道融合**：InsightFace ArcFace (主) + 对齐 AdaFace (辅)，加权融合识别  
✅ **自适应阈值**：根据 top1/top2 差距动态调整，减少误判  
✅ **实时签到**：D455 摄像头 + 产品级 PyQt5 UI，22人双通道特征库  
✅ **边缘部署**：鲁班猫 BTB（RK3588）+ RKNNLite2 + PySide6，支持 `det_10g.rknn / w600k_r50.rknn / adaface_ir50.rknn`  
✅ **分离式架构**：训练和测试完全独立，训练一次可多次测试  
✅ **高效灵活**：支持不同测试模式（正面/侧面/两者）快速切换  
✅ **兼容性强**：自动处理标签映射，支持新旧模型格式  
✅ **易于扩展**：模块化设计，方便添加新数据集或修改算法  

---

## 快速开始

### 环境要求

#### Windows / PC 训练与实时识别版本

- Python 3.10 (推荐 conda 环境 `2d3dadvface`)
- PyTorch 2.4+ with CUDA 12.1
- Intel RealSense D455 (实时签到系统)
- NVIDIA GPU (推荐，用于 GPU 加速)
- PyQt5 (实时签到 UI)
- 依赖包：torchvision, PIL, tqdm, numpy, opencv-python, pyrealsense2, insightface, onnxruntime

#### 鲁班猫 BTB / RK3588 边缘部署版本

- LubanCat BTB（RK3588）
- Ubuntu 20.04
- Python 3.10
- PySide6
- RKNNLite2 / rknn-toolkit-lite2
- Intel RealSense D455
- 依赖包：numpy, opencv-python, pillow, pyrealsense2, rknn-toolkit-lite2

### 安装依赖

```bash
# Windows / PC 版本完整环境安装
conda create -n 2d3dadvface python=3.10
conda activate 2d3dadvface
pip install -r requirements.txt
```

```bash
# 鲁班猫 BTB / RK3588 边缘部署版本
conda create -n 2d3dadvface python=3.10
conda activate 2d3dadvface
conda install -c conda-forge pyside6
pip install numpy opencv-python pillow pyrealsense2 rknn-toolkit-lite2
```

### 一键运行（推荐）

#### Windows PowerShell

```powershell
cd src/scripts

# 设置执行策略（首次运行需要）
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 第一步：训练模型并建立特征库（约35分钟）
.\run_train.ps1

# 第二步：测试所有模式（约6分钟）
.\run_all_tests.ps1
```

#### 命令行方式

```bash
cd src/scripts

# 训练
python train_and_build_gallery.py --cfp-root "../../cfp-dataset" --epochs 30

# 测试
python test_recognition.py --occluded-root "../../occluded" --mode both
```

---

## 项目结构

```
2d3dadvface/
├── README.md
├── requirements.txt
├── AdaFace-master/                # AdaFace 源码（IR-50 backbone 等）
│   ├── config.py
│   ├── convert.py
│   ├── data.py
│   ├── evaluate_utils.py
│   ├── head.py
│   ├── inference.py
│   ├── main.py
│   ├── net.py
│   └── utils.py
├── Dataset/                        # 实时识别人员头像数据集（按人目录组织）
│   ├── 001/
│   ├── 003/
│   └── ...
├── docs/                           # 项目文档与指南
│   ├── ADAFACE_REALTIME_GUIDE.md
│   └── ...
├── galleries/                      # 特征库（ArcFace / AdaFace / 元数据）
│   ├── gallery_adaface.npz
│   ├── gallery_arcface.npz
│   ├── gallery_features.pth
│   └── gallery_meta.json
├── models/                         # 模型文件与转换产物
│   ├── adaface_ir50_ms1mv2.ckpt
│   ├── adaface_ir50_ms1mv2.onnx
│   ├── adaface_ir50.rknn
│   ├── w600k_r50.rknn
│   └── det_10g.rknn
├── src/                            # 源代码目录
│   ├── datasets/                   # 数据加载器
│   │   ├── cfp_dataset_loader.py
│   │   └── occluded_loader.py
│   ├── models/                     # 模型定义与训练相关代码
│   ├── scripts/                    # 训练、测试、提取特征与 UI 脚本
│   │   ├── cfp_oldcode/             # 兼容旧版训练/测试脚本
│   │   │   ├── train_and_build_gallery.py
│   │   │   └── test_recognition.py
│   │   ├── extract_gallery_features_v2.py
│   │   └── ui/                     # 界面脚本（PyQt5 / RK3588 适配）
│   │       ├── ui_sign.py
│   │       ├── ui_sign_v2.py
│   │       └── ui_sign_v2_rk3588.py
│   └── utils/                      # 工具与识别封装
│       ├── adaface_recognizer.py
│       ├── face_detector.py
│       ├── face_processing.py
│       └── deploy/                 # 模型转换与部署脚本
│           ├── convert_to_onnx.py
│           └── convert_to_rknn.py
├── temp_results/                   # 截图与临时输出
├── cfp-dataset/                    # CFP 训练数据（外部数据集目录示例）
│   └── Data/Images/
└── occluded/                       # 遮挡（口罩）测试集示例
      └── 001-499/

```
---

## 实时签到系统

### 概述

基于 Intel RealSense D455 深度相机的人脸实时签到系统。**v4.0 引入 5 点关键点对齐 + InsightFace ArcFace / AdaFace 双通道融合识别管线**，提供产品级 PyQt5 桌面界面；**v5.1 新增鲁班猫 BTB（RK3588）+ RKNNLite2 + PySide6 边缘部署界面**。

### 运行

```powershell
conda activate 2d3dadvface

# v2 推荐 (5点对齐 + 双通道融合, 准确度更高)
python -u src/scripts/ui/ui_sign_v2.py

# v1 兼容 (bbox裁剪 + AdaFace单通道)
python -u src/scripts/ui/ui_sign.py

# 鲁班猫 BTB / RK3588 版本 (PySide6 / RKNN 适配)
python -u src/scripts/ui/ui_sign_v2_rk3588.py
```

### 识别管线 v2 (推荐)

```
D455 摄像头 → BGR帧 → InsightFace检测 + 5点关键点
    → ArcFace嵌入 (512d, 已对齐) ─┐
    → 5点相似变换对齐 → AdaFace (512d) ─┤
       └─ 加权融合 + 自适应阈值 → 匹配 → 签入 ✨
```

**v2 vs v1 对比**：

| 维度 | v1 (原版) | v2 (推荐) |
|------|----------|----------|
| 对齐方式 | bbox裁剪 → resize | 5点相似变换 → ArcFace标准模板 |
| 特征来源 | AdaFace 单通道 | ArcFace(主) + 对齐AdaFace(辅) |
| 融合策略 | 无 | 加权融合 + 双通道一致性加分 |
| 阈值类型 | 固定 | 自适应 (top1/top2 margin 动态) |
| 类内相似度 | ~0.35 | **0.49-0.78** |
| 默认阈值 | 0.30 | **0.45** |

### 界面布局

```
┌──────────────────────┬──────────────────────────────┐
│  Sign-In Manager v2  │  Live Recognition Feed ● Run │
│──────────────────────│──────────────────────────────│
│  [Tot:22] [Sig:0] [Rem:22] │                        │
│──────────────────────│    D455 Real-time Feed       │
│  Person List (22)     │    (640x480 @ 30fps)        │
│  ┌──────────────────┐ │    5-Point Aligned +        │
│  │ [张] 张三   ○    │ │    Dual-Channel Fusion      │
│  │ [李] 李四   ✓    │ │    A:0.65 D:0.58           │
│  │ ...              │ │                            │
│  └──────────────────┘ │                            │
│──────────────────────│──────────────────────────────│
│ [Reset] [Export]      │ FPS:28 Th[===●===]0.45 [Stop][Shot] │
└──────────────────────┴──────────────────────────────┘
```

### 功能特性

| 功能 | 说明 |
|------|------|
| 实时检测 | InsightFace (buffalo_l) 人脸检测 + 5点关键点, 640x640 |
| 人脸对齐 | 5点相似变换 → ArcFace 112x112 标准模板 |
| 双通道识别 | ArcFace w600k_r50 (主) + AdaFace IR-50 (辅), 加权融合 |
| 自适应阈值 | top1/top2 margin 动态调整，减少误判 |
| 特征库 | 22人双通道预建特征库 (gallery_arcface.npz + gallery_adaface.npz) |
| 签入管理 | 30帧稳定性确认、自动去重、签到时间追踪、CSV导出 |
| 阈值调节 | 滑块实时调整识别阈值 (0.05-0.80) |
| 暗色主题 | 深黑底 + 青绿 accent 配色，产品级视觉 |
| 边缘部署 | 鲁班猫 BTB / RK3588 / RKNNLite2 / PySide6 |
| NPU模型 | det_10g.rknn + w600k_r50.rknn + adaface_ir50.rknn |
| 优化策略 | SCRFD检测降频 + AdaFace临界分数复核，兼顾速度与精度 |

### 构建/更新特征库

```powershell
conda activate 2d3dadvface

# v2 推荐 (5点对齐 + 双通道)
python -u src/scripts/extract_gallery_features_v2.py

# v1 兼容
python -u src/scripts/extract_gallery_features_realtime.py
```

将 Dataset 目录下的头像图片提取特征存入 `galleries/`。v2 生成 `gallery_arcface.npz` + `gallery_adaface.npz` + `gallery_meta.json`。

---

## 数据集说明

### CFP 数据集（训练集）

- **人员数量**：500 人
- **图像数量**：每人 10 张（共 5004 张）
- **视图类型**：正面（frontal）+ 侧面（profile）
- **用途**：训练 ArcFace 模型 + 建立特征库
- **路径**：`cfp-dataset/Data/Images/{person_id}/{view}/`

### Occluded 数据集（测试集）

- **人员数量**：499 人（与 CFP 有重叠）
- **图像数量**：每人 30 张（共 14982 张）
- **视图类型**：正面 15 张 + 侧面 15 张
- **遮挡情况**：佩戴口罩
- **用途**：测试模型在遮挡场景下的识别能力
- **路径**：`occluded/{person_id}/{view}/`

---

## 使用方法

### 方法一：实时签到系统 v2 ⭐

```powershell
conda activate 2d3dadvface
python -u src/scripts/ui/ui_sign_v2.py
```

### 方法二：鲁班猫 BTB / RK3588 边缘部署版本 ⭐

```bash
conda activate 2d3dadvface

# RK3588 版本（PySide6 + RKNNLite2）
python -u src/scripts/ui/ui_sign_v2_rk3588.py
```

模型文件需放置在：

```text
models/
├── det_10g.rknn
├── w600k_r50.rknn
└── adaface_ir50.rknn
```

### 方法三：命令行实时识别 v2

```powershell
conda activate 2d3dadvface
python -u src/scripts/realtime_face_recognition_v2.py
```

快捷键：`Q` 退出 / `T` 提高阈值 / `G` 降低阈值 / `S` 截图

### 方法四：旧版系统 (兼容)

```powershell
# 旧版签到
python -u src/scripts/ui/ui_sign.py

# 旧版命令行识别
python -u src/scripts/realtime_face_recognition_d455.py
```

### 方法五：使用便捷脚本（CFP训练+测试）

#### 1. 训练模型

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
cd src
.\run_train.ps1
```

**输出文件**：
- `checkpoints/best_model.pth` - 训练好的模型
- `gallery_features.pth` - 500 人的特征库

#### 2. 测试识别

```powershell
# 测试正面视图
.\run_test.ps1 -Mode frontal

# 测试侧面视图
.\run_test.ps1 -Mode profile

# 测试两种视图
.\run_test.ps1 -Mode both

# 一次性测试所有模式
.\run_all_tests.ps1
```

### 方法六：直接使用 Python 命令（CFP训练+测试）

#### 1. 训练模型

```bash
cd src/scripts

python train_and_build_gallery.py \
    --cfp-root "../../cfp-dataset" \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.01 \
    --save-dir ../../models \
    --gallery-path ../../galleries/gallery_features.pth \
    --gpu 0
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cfp-root` | 必需 | CFP 数据集路径 |
| `--epochs` | 30 | 训练轮数 |
| `--batch-size` | 32 | 批次大小 |
| `--lr` | 0.01 | 学习率 |
| `--save-dir` | ./checkpoints | 模型保存目录 |
| `--gallery-path` | gallery_features.pth | 特征库保存路径 |
| `--gpu` | 0 | GPU ID |

#### 2. 测试识别

```bash
cd src/scripts

python test_recognition.py \
    --model-path ../../models/best_model.pth \
    --gallery-path ../../galleries/gallery_features.pth \
    --occluded-root "../../occluded" \
    --mode frontal \
    --gpu 0
```

**参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | ../../models/best_model.pth | 模型路径 |
| `--gallery-path` | ../../galleries/gallery_features.pth | 特征库路径 |
| `--occluded-root` | 必需 | 遮挡数据集路径 |
| `--mode` | both | 测试模式：`frontal` / `profile` / `both` |
| `--gpu` | 0 | GPU ID |

---

## 技术细节

### 工作流程

```
┌─────────────────────────────────────────────────┐
│           Phase 1: 训练模型                      │
│  CFP Dataset → ArcFace Model → best_model.pth  │
│  (500人, 5004张图像)    (ResNet50 + ArcFace)    │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Phase 2: 建立特征库                      │
│  CFP Dataset + Model → gallery_features.pth    │
│  (提取500人平均embedding)    (512维特征向量)      │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Phase 3: 测试识别                        │
│  Occluded Dataset + Gallery → Accuracy Report  │
│  (余弦相似度匹配)          (Top-1 准确率)         │
└─────────────────────────────────────────────────┘
```

### 核心技术

#### 1. AdaFace 实时识别

- **模型**：IR-50 Backbone，预训练于 MS1MV2
- **特征**：512 维 embedding，L2 归一化
- **匹配**：余弦相似度，阈值可调（默认 0.30）
- **检测**：InsightFace buffalo_l (SCRFD 人脸检测)

#### 2. ArcFace 损失函数

- **类型**：Additive Angular Margin Loss
- **作用**：在角度空间增加类别间距离，提升特征判别性
- **参数**：margin=0.5, scale=64

#### 3. 标签映射机制

由于 ArcFace 要求类别索引从 0 开始连续，而数据集中人员 ID 为 1-500，系统采用三遍扫描策略：

1. **第一遍**：收集所有图像和原始人员 ID
2. **第二遍**：只为实际有图像的人员创建标签映射（0-indexed）
3. **第三遍**：应用映射生成最终数据集

**反向映射**：建库时将映射标签转换回原始人员 ID，确保与测试集一致。

#### 4. 特征提取与匹配

- **提取**：从 ResNet50 backbone 获取 512 维 embedding
- **聚合**：对同一人的多张图像取平均作为该人特征
- **匹配**：使用余弦相似度计算测试图像与特征库中所有人的相似度
- **决策**：选择相似度最高的人作为识别结果

### 模型架构

```
Input Image (112×112×3)
        ↓
   ResNet50 Backbone
        ↓
 Global Average Pooling
        ↓
   Bottleneck Layer (FC)
        ↓
  Embedding (512-dim, L2 normalized)
        ↓
   ArcFace Head (training only)
        ↓
   Logits (num_classes)
```

---


## RK3588 / 鲁班猫BTB边缘部署版本

### 概述

在 v4.0（5点关键点对齐 + ArcFace/AdaFace 双通道融合）的基础上，项目新增基于 RK3588 NPU 的边缘部署版本。

适用环境：

- LubanCat BTB（RK3588）
- Ubuntu 20.04
- Intel RealSense D455
- RKNNLite2
- PySide6

该版本实现 PC 端训练、模型转换、边缘端部署的完整流程。

### RKNN模型说明

| 模型 | 作用 | 说明 |
|------|------|------|
| `det_10g.rknn` | 人脸检测 | SCRFD检测模型，输出人脸框和5点关键点 |
| `w600k_r50.rknn` | 主识别 | ArcFace主特征提取模型 |
| `adaface_ir50.rknn` | 辅助复核 | AdaFace辅助特征提取模型，用于临界样本复核 |

### PySide6界面

由于部分 ARM64 平台使用 pip 安装 PyQt5 容易出现编译失败或依赖冲突，鲁班猫 BTB 版本使用 PySide6 替代 PyQt5。

PySide6 版本保留：

- 左侧人员列表
- 右侧实时识别画面
- 自动签到
- 重置签到
- 导出CSV
- 阈值滑动调节
- FPS显示
- 中文显示
- 暗色主题

### RKNN识别流程

```text
D455摄像头
      ↓
det_10g.rknn
      ↓
5点关键点定位
      ↓
112×112人脸对齐
      ↓
w600k_r50.rknn
      ↓
高置信度直接确认

临界分数样本
      ↓
adaface_ir50.rknn
      ↓
双通道融合
      ↓
签到确认
```

### RK3588 / 鲁班猫 优化策略说明

RK3588 版本保持检测降频与按需复核的策略（检测不必每帧执行；ArcFace 主通道，AdaFace 仅在临界分数触发），以在精度与速度间取得平衡。对应脚本位于 `src/scripts/ui/ui_sign_v2_rk3588.py`。

### 环境安装

```bash
conda create -n 2d3dadvface python=3.10
conda activate 2d3dadvface

conda install -c conda-forge pyside6

pip install numpy opencv-python pillow pyrealsense2 rknn-toolkit-lite2
```

### 运行命令

```bash
# RK3588 运行（PySide6 + RKNNLite2）
python3 -u src/scripts/ui/ui_sign_v2_rk3588.py
```

---

## 常见问题

### Q1: PowerShell 无法运行 .ps1 脚本？

**A**: 需要先设置执行策略：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

或者永久允许当前用户：

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Q2: 显存不足怎么办？

**A**: 减小 batch size：

```bash
python train_and_build_gallery.py --batch-size 16 --cfp-root "../cfp-dataset"
```

### Q3: 如何提高识别准确率？

**A**: 
1. 增加训练轮数：`--epochs 50`
2. 调整学习率：`--lr 0.001`
3. 使用更多训练数据：修改代码使用 `mode='both'`
4. 尝试更强的 backbone（如 ResNet101）

### Q4: 测试时出现 KeyError: 'num_classes'？

**A**: 这是旧版本模型的兼容性问题。新版测试代码已自动修复，会从模型权重中推断类别数。如果仍遇到问题，请重新训练模型。

### Q5: 可以测试自定义数据集吗？

**A**: 可以！只需按照 occluded 数据集的格式组织图像：

```
custom_dataset/
├── 001/
│   ├── frontal/
│   │   ├── img1.jpg
│   │   └── ...
│   └── profile/
│       ├── img1.jpg
│       └── ...
├── 002/
│   └── ...
```

然后运行：

```bash
python test_recognition.py --occluded-root "./custom_dataset" --mode both
```

### Q6: 训练中断了怎么办？

**A**: 目前不支持断点续训。建议：
1. 减少 epochs 分批训练
2. 或使用更稳定的训练环境

### Q7: 为什么只有 499 人而不是 500 人？

**A**: Occluded 数据集中只有 499 人，但 CFP 数据集和特征库包含 500 人。测试时会自动跳过不在特征库中的人员。

### Q8: D455 摄像头无法打开？

**A**:
1. 确认 D455 连接到 USB 3.0 接口
2. 关闭 Intel RealSense Viewer、浏览器等占用设备的程序
3. 用 Intel RealSense Viewer 检查彩色流是否正常
4. 重新插拔或重启电脑

### Q9: 实时签到识别不准确？

**A**:
1. **使用 v2 系统**：`python -u src/scripts/ui/ui_sign_v2.py`（5点对齐+双通道，准确度大幅提升）
2. 调整阈值滑块（降低阈值 = 更宽松的匹配）
3. 确保光照条件良好，人脸正对摄像头
4. 重新运行 `extract_gallery_features_v2.py` 更新特征库
5. 确认识别者的头像已在 Dataset 目录中，每人建议 5 张不同角度照片

### Q10: 如何添加新人员？

**A**:
1. 将新人员头像放入 `Dataset/` 目录（新建文件夹，放 5 张不同角度照片）
2. 运行 `python -u src/scripts/extract_gallery_features_v2.py` 重建双通道特征库
3. 重启签到系统即可


### Q11: 鲁班猫 BTB 版本需要哪些 RKNN 模型？

**A**: 需要将以下三个模型放入 `models/` 目录：

```text
det_10g.rknn
w600k_r50.rknn
adaface_ir50.rknn
```

其中 `det_10g.rknn` 用于人脸检测，`w600k_r50.rknn` 用于 ArcFace 主识别，`adaface_ir50.rknn` 用于 AdaFace 辅助复核。

### Q12: PySide6 和 PyQt5 版本有什么区别？

**A**: PC 端原版使用 PyQt5；鲁班猫 BTB / RK3588 边缘部署版本使用 PySide6。PySide6 在 ARM64 和 Qt6 环境中兼容性更好，界面功能保持一致。

### Q13: 优化版会不会降低识别精度？

**A**: 优化版不会简单关闭 AdaFace，而是采用“ArcFace主识别 + AdaFace临界复核”的策略。检测降频和界面渲染优化主要提升速度；AdaFace仍会在不确定样本中参与判断，因此比单通道识别更稳。


---

## 性能指标

### 训练性能

- **训练准确率**：~99%（30 epochs）
- **训练时间**：约 30 分钟（GPU）
- **建库时间**：约 5 分钟（GPU）

### 测试性能（示例）

| 测试模式 | 样本数 | Top-1 准确率 | 测试时间 |
|---------|--------|-------------|---------|
| Frontal | 14,982 | ~91% | ~25 秒 |
| Profile | 14,982 | ~85% | ~25 秒 |
| Both | 29,964 | ~88% | ~50 秒 |

*注：具体准确率取决于训练收敛情况和测试数据分布*

### 效率对比

**传统方式**（每次测试都重新训练）：
- 测试 3 种模式：3 × (30 + 5 + 2) = **111 分钟**

**新方式**（训练一次，多次测试）：
- 首次：30 + 5 = 35 分钟
- 后续每次测试：~2 分钟
- 测试 3 种模式：35 + 3 × 2 = **41 分钟**
- **节省时间：63%** ⚡


### 鲁班猫BTB边缘部署性能

| 模式 | 推理策略 | 说明 |
|------|----------|------|
| RKNN标准版 | 每帧检测 + ArcFace + AdaFace | 精度优先，速度相对较低 |
| RKNN优化版 | 检测降频 + ArcFace主识别 + AdaFace临界复核 | 推荐部署方案，兼顾速度与精度 |

优化版核心思想：

- 检测不必每帧执行
- ArcFace作为主识别通道
- AdaFace只在临界分数样本中复核
- 签到依旧采用多帧稳定确认，避免单帧误识别


---

## 故障排除

### 错误：ModuleNotFoundError

```bash
pip install torch torchvision pillow tqdm numpy
```

### 错误：CUDA out of memory

减小 batch size 或使用 CPU：

```bash
python train_and_build_gallery.py --batch-size 16 --cfp-root "../cfp-dataset"
# 或不指定 --gpu 参数使用 CPU
```

### 错误：Dataset not found

检查数据集路径是否正确：

```bash
ls cfp-dataset/Data/Images/
ls occluded/
```

### 准确率为 0% 或很低

1. 确认模型已正确训练（train acc > 90%）
2. 确认特征库已正确建立（500 人）
3. 检查测试集标签是否与特征库 ID 匹配
4. 尝试增加训练轮数

---

## 开发历史

### v5.1（当前开发版）

- ✅ 鲁班猫 BTB（RK3588）边缘部署适配
- ✅ RKNNLite2 板端推理
- ✅ `det_10g.rknn` SCRFD人脸检测
- ✅ `w600k_r50.rknn` ArcFace主识别
- ✅ `adaface_ir50.rknn` AdaFace临界复核
- ✅ PySide6 签到界面
- ✅ SCRFD检测降频优化
- ✅ AdaFace按需调用，兼顾速度与精度

### v4.0

- ✅ **5点关键点人脸对齐**：cv2.estimateAffinePartial2D 相似变换 → ArcFace 112x112 标准模板
- ✅ **双通道融合识别**：InsightFace ArcFace w600k_r50 (主) + 对齐 AdaFace IR-50 (辅)
- ✅ **自适应阈值**：top1/top2 margin 动态调整，减少误判
- ✅ **质量过滤**：det_score < 0.7 自动拒绝 + 类间混淆检测
- ✅ 类内相似度从 ~0.35 提升至 **0.49-0.78**
- ✅ v2 签到 UI (ui_sign_v2.py)，双通道得分可视化

### v3.0

- ✅ 集成 AdaFace IR-50 预训练模型
- ✅ Intel RealSense D455 实时识别管线
- ✅ 产品级 PyQt5 签到系统 UI (FaceSign)
- ✅ 21人实时特征库
- ✅ 签入管理（去重、时间追踪、CSV导出）
- ✅ CFP-500 PyQt5 测试界面

### v2.0

- ✅ 分离训练和测试脚本
- ✅ 添加 PowerShell 快捷脚本
- ✅ 修复标签映射问题
- ✅ 修复 gallery 维度问题
- ✅ 支持新旧模型格式兼容
- ✅ 完善文档和错误处理

### v1.0

- 初始版本：单脚本实现训练+建库+测试
- 基础 ArcFace 模型
- CFP 数据集支持

---

## 引用

如果使用本项目的代码或方法，请引用：

```bibtex
@inproceedings{sengupta2016frontal,
  title={Frontal to profile face verification in the wild},
  author={Sengupta, Soumyadip and Chen, Jun-Cheng and Castillo, Carlos and Patel, Vishal M and Chellappa, Rama and Jacobs, David W},
  booktitle={2016 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={1--9},
  year={2016},
  organization={IEEE}
}

@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}
```

---

## 许可证

本项目仅供学术研究使用。

---

## 联系方式

如有问题或建议，欢迎提交 Issue！

---

**祝研究顺利！** 🚀

# Face Recognition System for Access Control
## 门禁人脸识别系统

基于 ArcFace 的人脸识别系统，用于在遮挡（口罩）场景下进行人脸识别测试。

---

## 📋 目录

- [项目简介](#项目简介)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [数据集说明](#数据集说明)
- [使用方法](#使用方法)
- [技术细节](#技术细节)
- [常见问题](#常见问题)
- [性能指标](#性能指标)

---

## 项目简介

本系统实现了一个完整的人脸识别流程：

1. **训练阶段**：使用 CFP 数据集训练 ArcFace 模型
2. **建库阶段**：从 CFP 数据集提取 500 人的特征向量建立特征库
3. **测试阶段**：在遮挡数据集（occluded）上测试识别准确率

### 核心特点

✅ **分离式架构**：训练和测试完全独立，训练一次可多次测试  
✅ **高效灵活**：支持不同测试模式（正面/侧面/两者）快速切换  
✅ **兼容性强**：自动处理标签映射，支持新旧模型格式  
✅ **易于扩展**：模块化设计，方便添加新数据集或修改算法  

---

## 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.9+
- CUDA (推荐，用于 GPU 加速)
- 依赖包：torchvision, PIL, tqdm, numpy

### 安装依赖

```bash
pip install torch torchvision pillow tqdm numpy
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
├── README.md                          # 项目说明文档（本文件）
├── requirements.txt                   # Python 依赖包
├── models/                            # 训练好的模型（自动生成）
│   └── best_model.pth                 # ArcFace 模型检查点
├── galleries/                         # 特征库（自动生成）
│   └── gallery_features.pth           # 500人特征向量
├── src/                               # 源代码目录
│   ├── README.md                      # src 目录说明
│   ├── models/                        # 模型定义
│   │   ├── __init__.py
│   │   └── arcface_model.py          # ArcFace ResNet50 模型
│   ├── datasets/                      # 数据集加载器
│   │   ├── __init__.py
│   │   ├── cfp_dataset_loader.py     # CFP 数据集加载器
│   │   └── occluded_loader.py        # 遮挡数据集加载器
│   ├── scripts/                       # 主脚本
│   │   ├── train_and_build_gallery.py    # ⭐ 训练脚本
│   │   ├── test_recognition.py           # ⭐ 测试脚本
│   │   ├── face_recognition_system.py    # 旧版整合脚本（参考）
│   │   ├── run_train.ps1                 # PowerShell: 训练快捷方式
│   │   ├── run_test.ps1                  # PowerShell: 测试快捷方式
│   │   └── run_all_tests.ps1             # PowerShell: 全部测试
│   └── utils/                         # 工具函数（预留）
├── cfp-dataset/                       # CFP 人脸数据集
│   └── Data/Images/                   # 500人 × 10张图像
└── occluded/                          # 遮挡（口罩）数据集
    └── 001-499/                       # 499人 × 30张图像（正侧各15张）
```

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

### 方法一：使用便捷脚本（推荐）⭐

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

### 方法二：直接使用 Python 命令

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

#### 1. ArcFace 损失函数

- **类型**：Additive Angular Margin Loss
- **作用**：在角度空间增加类别间距离，提升特征判别性
- **参数**：margin=0.5, scale=64

#### 2. 标签映射机制

由于 ArcFace 要求类别索引从 0 开始连续，而数据集中人员 ID 为 1-500，系统采用三遍扫描策略：

1. **第一遍**：收集所有图像和原始人员 ID
2. **第二遍**：只为实际有图像的人员创建标签映射（0-indexed）
3. **第三遍**：应用映射生成最终数据集

**反向映射**：建库时将映射标签转换回原始人员 ID，确保与测试集一致。

#### 3. 特征提取与匹配

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

### v2.0（当前版本）

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

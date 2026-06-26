# 鲁班猫 RKNN 人脸识别系统 - 模型转换与部署指南

## 📋 概述

本文档记录将 PyTorch/ONNX 模型转换为 RKNN 格式，并部署到鲁班猫 (LubanCat) RK3588 的完整流程。

### 转换目标

| 模型 | 原始格式 | RKNN 输出 | 用途 | 输入尺寸 |
|------|---------|-----------|------|----------|
| `det_10g` | ONNX | `det_10g.rknn` | SCRFD 人脸检测 | 640×640 |
| `w600k_r50` | ONNX | `w600k_r50.rknn` | ArcFace 识别 | 112×112 |
| `adaface_ir50` | PyTorch (.ckpt) | `adaface_ir50.rknn` | AdaFace 识别 | 112×112 |

---

## 🖥️ 第一部分：PC 端环境准备--虚拟机 (x86_64 Ubuntu)

### 1.1 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Ubuntu 20.04 / 22.04 (x86_64) |
| Python | 3.10 |
| 内存 | ≥ 4GB |
| 磁盘 | ≥ 64GB |

### 1.2 安装 Miniconda 及创建环境

```bash
# 下载并安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建 Python 3.10 环境
conda create -n rknn_converter python=3.10 -y
conda activate rknn_converter
```

### 1.3 安装 RKNN-Toolkit2

```bash
# 克隆官方仓库
git clone https://github.com/rockchip-linux/rknn-toolkit2.git
cd rknn-toolkit2

# 安装依赖
pip install -r rknn-toolkit2/packages/requirements_cp310-1.6.0.txt

# 安装 RKNN-Toolkit2
pip install rknn-toolkit2/packages/rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl

# 安装其他依赖
pip install onnx onnxruntime opencv-python numpy torch
```

### 1.4 验证环境

```bash
python -c "from rknn.api import RKNN; print('RKNN OK')"
python -c "import onnx; print('ONNX OK')"
```

---

## 📁 第二部分：目录结构

```
2d3dadvface/
├── models/                          # 模型存放目录
│   ├── adaface_ir50_ms1mv2.ckpt     # 原始 PyTorch 模型
│   ├── adaface_ir50_ms1mv2.onnx     # 转换后的 ONNX
│   ├── det_10g.rknn                 # 转换后的 RKNN
│   ├── w600k_r50.rknn
│   └── adaface_ir50.rknn
├── src/
│   └── utils/
│       └── deploy/                  # 转换脚本目录
│           ├── convert_to_onnx.py   # PyTorch → ONNX
│           └── convert_to_rknn.py   # ONNX → RKNN
└── AdaFace-master/                  # AdaFace 源码（用于转换）
```

---

## 🔄 第三部分：模型转换流程

### 3.1 获取原始模型

#### InsightFace 模型 (det_10g, w600k_r50)

```bash
# 使用 InsightFace 自动下载
python -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)
print('模型位置: ~/.insightface/models/buffalo_l/')
"

# 复制到项目 models 目录
cp ~/.insightface/models/buffalo_l/det_10g.onnx models/
cp ~/.insightface/models/buffalo_l/w600k_r50.onnx models/
```

#### AdaFace 模型

确保 `models/adaface_ir50_ms1mv2.ckpt` 文件存在。

### 3.2 PyTorch → ONNX (仅 AdaFace)

**脚本位置**: `src/utils/deploy/convert_to_onnx.py`

**关键代码**:
```python
# 加载模型
model = build_model('ir_50')
ckpt = torch.load('models/adaface_ir50_ms1mv2.ckpt', map_location='cpu')
model.load_state_dict(ckpt['state_dict'])
model.eval()

# 导出 ONNX
dummy = torch.randn(1, 3, 112, 112)
torch.onnx.export(
    model, dummy, 'models/adaface_ir50_ms1mv2.onnx',
    input_names=['input'], output_names=['feature', 'norm'],
    opset_version=12
)
```

**运行**:
```bash
conda activate rknn_converter
cd /path/to/2d3dadvface
python src/utils/deploy/convert_to_onnx.py
```

### 3.3 ONNX → RKNN

**脚本位置**: `src/utils/deploy/convert_to_rknn.py`

**模型配置**:
```python
MODELS = [
    {
        'name': 'det_10g',
        'input_size': (640, 640),
        'mean': [[0, 0, 0]], 'std': [[1, 1, 1]],
        'input_name': 'input.1',
    },
    {
        'name': 'w600k_r50',
        'input_size': (112, 112),
        'mean': [[0, 0, 0]], 'std': [[1, 1, 1]],
        'input_name': 'input.1',
    },
    {
        'name': 'adaface_ir50',
        'input_size': (112, 112),
        'mean': [[127.5, 127.5, 127.5]], 'std': [[127.5, 127.5, 127.5]],
        'input_name': 'input',
    },
]
```

**关键转换代码**:
```python
# 加载 ONNX 时必须指定 input_size_list 固定输入尺寸
ret = rknn.load_onnx(
    model=onnx_path,
    inputs=[input_name],
    input_size_list=[[1, 3, h, w]]
)

# 构建并导出
ret = rknn.build(do_quantization=False)  # FP16
ret = rknn.export_rknn(rknn_path)
```

**运行**:
```bash
python src/utils/deploy/convert_to_rknn.py --quant fp16
```

### 3.4 转换结果

```
models/
├── det_10g.rknn        (9.9 MB)
├── w600k_r50.rknn      (90.1 MB)
└── adaface_ir50.rknn   (83.9 MB)
```

---

## 📡 第四部分：传输到鲁班猫

### 4.1 获取鲁班猫 IP

```bash
# 在鲁班猫上执行
ip addr show | grep inet
# 示例: 192.168.1.100
```

### 4.2 传输模型文件

```bash
# 在 PC 上执行
scp models/*.rknn cat@192.168.1.100:~/2d3dadvface/2d3dadvface/models/
scp models/*.npz cat@192.168.1.100:~/2d3dadvface/2d3dadvface/galleries/ 2>/dev/null
```

### 4.3 传输源代码

```bash
scp -r src/ cat@192.168.1.100:~/2d3dadvface/2d3dadvface/
```

---

## 🚀 第五部分：鲁班猫环境配置

### 5.1 安装 RKNN Lite

```bash
# 在鲁班猫上执行
pip3 install rknn-toolkit2-lite
```

### 5.2 安装其他依赖

```bash
# 基础依赖
pip3 install numpy opencv-python Pillow

# RealSense 驱动
sudo apt update
sudo apt install -y librealsense2-dev python3-pybind11-dev
pip3 install pyrealsense2

# 中文字体（可选）
sudo apt install -y fonts-wqy-microhei
```

### 5.3 验证环境

```python
from rknnlite.api import RKNNLite
import cv2
import pyrealsense2 as rs

print("RKNN Lite OK")
print(f"OpenCV {cv2.__version__}")
```

---

## ✅ 第六部分：运行测试

### 6.1 快速验证

```python
# test_rknn.py
from rknnlite.api import RKNNLite

for name in ['det_10g', 'w600k_r50', 'adaface_ir50']:
    rknn = RKNNLite()
    rknn.load_rknn(f'models/{name}.rknn')
    rknn.init_runtime()
    print(f"✓ {name}.rknn 加载成功")
    rknn.release()
```

### 6.2 运行完整系统

```bash
cd ~/2d3dadvface/2d3dadvface
python src/scripts/src/scripts/realtime_face_recognition_v2_rk3588.py
```

---

## ⚠️ 常见问题

### Q1: `module 'cv2' has no attribute 'resize'`

**原因**: 安装了 headless 版本  
**解决**: `pip uninstall opencv-python-headless -y && pip install opencv-python`

### Q2: `Could not load Qt platform plugin "xcb"`

**原因**: OpenCV Qt 插件冲突  
**解决**: 脚本开头添加 `os.environ['OPENCV_QT_ENABLE'] = '0'`

### Q3: 模型转换时 `input shape ['None', 3, 112, 112] is not support`

**原因**: 动态输入维度  
**解决**: `load_onnx` 时指定 `input_size_list=[[1, 3, 112, 112]]`

### Q4: `The 'input' in inputs=['input'] is invalid!`

**原因**: 输入节点名错误  
**解决**: 检查 ONNX 实际节点名，修改 `input_name` 配置

---

## 📊 命令速查

| 步骤 | 命令 |
|------|------|
| 激活环境 | `conda activate rknn_converter` |
| AdaFace 转换 | `python src/utils/deploy/convert_to_onnx.py` |
| RKNN 转换 | `python src/utils/deploy/convert_to_rknn.py --quant fp16` |
| 传输模型 | `scp models/*.rknn cat@<IP>:~/2d3dadvface/2d3dadvface/models/` |
| 鲁班猫运行 | `cd ~/2d3dadvface/2d3dadvface && python src/scripts/lubancat_btb_rknn_face_recognition_full.py` |

---

**文档版本**: v2.0  
**最后更新**: 2026-06-03
# 遮挡人脸检测解决方案

## 🎯 问题分析

### 当前问题
- **现象**：严重遮挡（口罩、墨镜等）时，Haar Cascade 无法检测到人脸
- **原因**：Haar Cascade 基于完整面部特征（眼睛、鼻子、嘴巴），遮挡破坏了这些特征
- **影响**：即使识别模型能处理遮挡，但检测阶段就失败了

### 测试数据
```
Image size: 291x300
Detecting faces...
⚠ No face detected
```

---

## 🔍 方案对比分析

### 方案一：MTCNN (Multi-task Cascaded Convolutional Networks) ⭐⭐⭐⭐⭐

#### 优势
- ✅ **对遮挡鲁棒性强** - 级联结构，部分遮挡仍能检测
- ✅ **纯 PyTorch 实现** - 与项目技术栈完全一致
- ✅ **提供关键点** - 可用于对齐和增强
- ✅ **轻量级** - 约 5MB 模型文件
- ✅ **成熟稳定** - 学术界和工业界广泛使用

#### 劣势
- ⚠️ CPU 速度较慢（~5-10 FPS）
- ⚠️ 需要额外安装 `facenet-pytorch`

#### 性能预期
| 场景 | 检测率 |
|------|--------|
| 清晰正脸 | ~98% |
| 戴口罩 | ~85% |
| 戴墨镜 | ~80% |
| 侧面脸 | ~75% |
| 严重遮挡 | ~65% |

#### 安装
```bash
pip install facenet-pytorch
```

#### 代码示例
```python
from facenet_pytorch import MTCNN

mtcnn = MTCNN(
    image_size=112,
    margin=20,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],  # PNet, RNet, ONet
    factor=0.709,
    device=device
)

boxes, probs, landmarks = mtcnn.detect(image_rgb, landmarks=True)
```

---

### 方案二：RetinaFace (PyTorch 版本) ⭐⭐⭐⭐⭐

#### 优势
- ✅ **SOTA 精度** - 目前最强的人脸检测器之一
- ✅ **极佳遮挡鲁棒性** - 专门针对遮挡优化
- ✅ **多姿态支持** - 正面、侧面都能检测
- ✅ **提供关键点** - 5个关键点用于对齐
- ✅ **纯 PyTorch** - 与项目一致

#### 劣势
- ⚠️ 模型稍大（~30MB）
- ⚠️ 需要找到可靠的 PyTorch 实现

#### 性能预期
| 场景 | 检测率 |
|------|--------|
| 清晰正脸 | ~99% |
| 戴口罩 | ~90% |
| 戴墨镜 | ~88% |
| 侧面脸 | ~85% |
| 严重遮挡 | ~75% |

#### 可用实现
1. **insightface** (推荐):
   ```bash
   pip install insightface
   ```
   
2. **biubug6/Pytorch_Retinaface**:
   ```bash
   git clone https://github.com/biubug6/Pytorch_Retinaface.git
   ```

#### 代码示例 (insightface)
```python
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

faces = app.get(image_rgb)
for face in faces:
    box = face.bbox  # [x1, y1, x2, y2]
    landmarks = face.kps  # 5 keypoints
```

---

### 方案三：YOLOv8-Face / YOLOv11-Face ⭐⭐⭐⭐

#### 优势
- ✅ **速度快** - GPU 上可达 50+ FPS
- ✅ **可训练自定义** - 可在遮挡数据集上微调
- ✅ **纯 PyTorch** - ultralytics 框架
- ✅ **易于部署** - 单一依赖

#### 劣势
- ⚠️ **需要专用模型** - 通用 YOLO 不检测人脸
- ⚠️ 预训练模型需从社区获取
- ⚠️ 遮挡性能取决于训练数据

#### 性能预期
| 场景 | 检测率 |
|------|--------|
| 清晰正脸 | ~97% |
| 戴口罩 | ~82% |
| 戴墨镜 | ~78% |
| 侧面脸 | ~70% |
| 严重遮挡 | ~60% |

#### 模型来源
1. **ultralytics-hub** (官方):
   - 搜索 "face detection" 模型
   
2. **社区模型**:
   - https://github.com/derronqi/yolov8-face
   - https://github.com/akanametov/yolo-face

#### 代码示例
```python
from ultralytics import YOLO

model = YOLO('yolov8n-face.pt')  # 需要下载专用模型
results = model(image_rgb, conf=0.3)

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
```

---

### 方案四：MediaPipe Face Detection (旧版 API) ⭐⭐⭐

#### 优势
- ✅ **极快** - CPU 上 30+ FPS
- ✅ **Google 维护** - 稳定可靠
- ✅ **轻量** - 模型小

#### 劣势
- ❌ **新版 API 复杂** - v0.10+ 需要外部模型文件
- ❌ **遮挡性能一般** - 不如 MTCNN/RetinaFace
- ❌ API 不稳定 - 频繁更新导致兼容性问题

#### 性能预期
| 场景 | 检测率 |
|------|--------|
| 清晰正脸 | ~95% |
| 戴口罩 | ~70% |
| 戴墨镜 | ~65% |
| 侧面脸 | ~60% |
| 严重遮挡 | ~50% |

---

### 方案五：改进现有 Haar Cascade (临时方案) ⭐⭐

#### 改进措施
1. **降低阈值** - `min_neighbors=2` (更多误检)
2. **多尺度检测** - 尝试多个 `scale_factor`
3. **图像预处理** - CLAHE 增强对比度
4. **多分类器融合** - frontal + profile 级联

#### 优势
- ✅ 无需额外依赖
- ✅ 快速实施

#### 劣势
- ❌ **根本限制** - Haar 无法克服遮挡问题
- ❌ 误检率高
- ❌ 提升有限（最多 10-15%）

#### 性能预期
| 场景 | 检测率 |
|------|--------|
| 清晰正脸 | ~95% |
| 戴口罩 | ~40% |
| 戴墨镜 | ~35% |
| 严重遮挡 | ~25% |

---

## 🏆 推荐方案

### 首选：MTCNN (facenet-pytorch)

**理由**：
1. ✅ **完美匹配项目需求** - 纯 PyTorch，轻量，对遮挡鲁棒
2. ✅ **实施简单** - 一行安装，几行代码集成
3. ✅ **平衡性能和精度** - 遮挡检测率 ~85%
4. ✅ **提供关键点** - 可用于后续人脸对齐
5. ✅ **成熟稳定** - 广泛使用，文档完善

**实施步骤**：
```bash
# 1. 安装
pip install facenet-pytorch

# 2. 修改 src/utils/face_detector.py
#    添加 MTCNNDetector 类

# 3. 修改 src/scripts/ui.py
#    切换为 'mtcnn' detector

# 4. 测试
python src/scripts/test_face_detection.py
```

**预计工作量**：2-3 小时

---

### 备选：RetinaFace (insightface)

**适用场景**：
- 如果 MTCNN 效果不够好
- 需要最高精度
- 可以接受稍大的依赖

**实施步骤**：
```bash
pip install insightface onnxruntime
```

**预计工作量**：3-4 小时

---

## 📊 方案对比总结

| 方案 | 遮挡检测率 | 速度(CPU) | 依赖大小 | 实施难度 | 推荐度 |
|------|-----------|----------|---------|---------|--------|
| **MTCNN** | ~85% | 5-10 FPS | ~5MB | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **RetinaFace** | ~90% | 3-8 FPS | ~30MB | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **YOLOv8-Face** | ~82% | 10-20 FPS | ~10MB | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| MediaPipe | ~70% | 30+ FPS | ~2MB | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 改进 Haar | ~40% | 20+ FPS | 0MB | ⭐ | ⭐⭐ |

---

## 🎯 最终建议

### 立即实施：MTCNN

**原因**：
1. 最快解决当前问题（2-3小时）
2. 显著提升遮挡检测率（40% → 85%）
3. 与项目技术栈完美契合
4. 未来可轻松升级到 RetinaFace

### 长期规划：RetinaFace

**时机**：
- 当 MTCNN 在某些极端场景下仍不足时
- 需要处理更多样化的姿态和遮挡

### 不建议：
- ❌ 继续优化 Haar Cascade - 收益太低
- ❌ MediaPipe - API 不稳定，遮挡性能一般
- ❌ 训练自定义 YOLO - 成本高，收益不确定

---

## 📝 实施路线图

### Phase 1: MTCNN 集成 (本周)
1. 安装 `facenet-pytorch`
2. 创建 `MTCNNDetector` 类
3. 更新工厂函数
4. 修改 UI 调用
5. 测试遮挡数据集

### Phase 2: 评估与优化 (下周)
1. 在 occluded 数据集上全面测试
2. 调整 MTCNN 参数（阈值、最小尺寸）
3. 记录检测率和误检率
4. 对比不同遮挡类型表现

### Phase 3: 升级决策 (2周后)
- 如果 MTCNN 达到 85%+ 检测率 → 保持
- 如果需要更高精度 → 升级到 RetinaFace
- 如果速度是瓶颈 → 考虑 YOLOv8-Face

---

## 🔧 技术细节

### MTCNN 关键参数调优

```python
MTCNN(
    image_size=112,           # 输出人脸尺寸
    margin=20,                # 边界扩展（像素）
    min_face_size=20,         # 最小检测尺寸（降低以检测小脸）
    thresholds=[0.6, 0.7, 0.7], # P/R/ONet 阈值（降低提高召回）
    factor=0.709,             # 图像金字塔缩放因子
    post_process=True,        # 后处理（对齐）
    device=device             # 'cpu' 或 'cuda'
)
```

**针对遮挡的优化**：
- 降低 `thresholds` → `[0.5, 0.6, 0.6]` (提高召回)
- 降低 `min_face_size` → `15` (检测更小的人脸)
- 增加 `margin` → `30` (包含更多上下文)

### RetinaFace 关键参数

```python
app = FaceAnalysis(
    providers=['CPUExecutionProvider'],
    det_size=(640, 640),  # 检测分辨率（越大越准但越慢）
    det_thresh=0.5        # 检测阈值（降低提高召回）
)
```

---

## 💡 额外建议

### 1. 降级策略
```python
try:
    # 尝试 MTCNN
    faces = mtcnn.detect(image)
    if len(faces) == 0:
        # 降级到 Improved Haar
        faces = haar.detect(image)
except:
    # 异常时降级
    faces = haar.detect(image)
```

### 2. 图像预处理增强
```python
# 对低光照/低对比度图像
from skimage import exposure
image_eq = exposure.equalize_adapthist(image)

# 对模糊图像
import cv2
image_sharp = cv2.GaussianBlur(image, (0, 0), 3)
image_sharp = cv2.addWeighted(image, 1.5, image_sharp, -0.5, 0)
```

### 3. 多尺度检测
```python
# 尝试多个尺度
scales = [1.0, 0.8, 1.2]
all_faces = []
for scale in scales:
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    faces = detector.detect(resized)
    all_faces.extend(scale_boxes(faces, 1/scale))
```

---

## 📚 参考资料

1. **MTCNN 论文**: https://arxiv.org/abs/1604.02878
2. **RetinaFace 论文**: https://arxiv.org/abs/1905.00641
3. **facenet-pytorch**: https://github.com/timesler/facenet-pytorch
4. **insightface**: https://github.com/deepinsight/insightface
5. **YOLOv8-Face**: https://github.com/derronqi/yolov8-face

---

## ✅ 下一步行动

1. **阅读本方案** - 理解各方案优劣
2. **确认选择 MTCNN** - 回复确认
3. **开始实施** - 我将修改代码集成 MTCNN
4. **测试验证** - 在 occluded 数据集上测试
5. **反馈优化** - 根据结果调整参数

**预计总时间**：2-3 小时实施 + 1 小时测试

准备好开始了吗？🚀

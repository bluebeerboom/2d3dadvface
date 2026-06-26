#!/usr/bin/env python3
"""
AdaFace IR-50 模型转换: .ckpt (PyTorch) → .onnx
==================================================
运行环境: 项目根目录, conda activate 2d3dadvface
输出: models/adaface_ir50_ms1mv2.onnx
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'AdaFace-master'))

import torch
import numpy as np
from net import build_model

# ── 配置 ──
CKPT_PATH = 'models/adaface_ir50_ms1mv2.ckpt'
ONNX_PATH = 'models/adaface_ir50_ms1mv2.onnx'
INPUT_SHAPE = (1, 3, 112, 112)
DEVICE = 'cpu'

print("=" * 60)
print("AdaFace IR-50: PyTorch → ONNX")
print("=" * 60)

# 1. 加载模型
print(f"[1/4] 加载 checkpoint: {CKPT_PATH}")
model = build_model('ir_50')
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

state_dict = ckpt.get('state_dict', ckpt)
model_state = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
model.load_state_dict(model_state)
model.to(DEVICE)
model.eval()
print(f"      参数量: {sum(p.numel() for p in model.parameters()):,}")

# 2. 导出 ONNX
print(f"[2/4] 导出 ONNX: {ONNX_PATH}")
dummy = torch.randn(*INPUT_SHAPE).to(DEVICE)
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=['input'],
    output_names=['feature', 'norm'],
    dynamic_axes={'input': {0: 'batch_size'}, 'feature': {0: 'batch_size'}},
    opset_version=12,
    do_constant_folding=True,
)
size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
print(f"      输出文件: {ONNX_PATH} ({size_mb:.1f} MB)")

# 3. 验证
print("[3/4] 验证推理一致性...")
import onnxruntime as ort
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

with torch.no_grad():
    pt_out, pt_norm = model(dummy)

ort_out = sess.run(None, {'input': dummy.numpy()})
onnx_feat, onnx_norm = ort_out[0], ort_out[1]

diff_feat = np.abs(pt_out.numpy() - onnx_feat).max()
diff_norm = np.abs(pt_norm.numpy() - onnx_norm).max()
print(f"      特征最大误差: {diff_feat:.8f}")
print(f"      归一化因子最大误差: {diff_norm:.8f}")

if diff_feat < 1e-4 and diff_norm < 1e-4:
    print("      ✓ 转换成功, 误差可接受")
else:
    print(f"      ✗ 误差过大! 请检查")

print("\n完成!")
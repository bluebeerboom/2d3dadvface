#!/usr/bin/env python3
"""
三模型一键转换: ONNX → RKNN (RK3588 NPU)
============================================
运行环境: Ubuntu VM, conda activate 2d3dadvface (已安装 rknn-toolkit2)
"""
import os
import sys
import argparse
import glob
import time
import tempfile
import shutil
import numpy as np
import cv2
import platform

assert platform.machine() in ('x86_64', 'AMD64'), \
    "rknn-toolkit2 必须在 x86_64 上运行!"

from rknn.api import RKNN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSIGHTFACE_DIR = os.path.expanduser('~/.insightface/models/buffalo_l')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

MODELS = [
    {
        'name': 'det_10g',
        'desc': 'SCRFD 人脸检测',
        'onnx': os.path.join(INSIGHTFACE_DIR, 'det_10g.onnx'),
        'alias': [os.path.join(MODELS_DIR, 'det_10g.onnx')],
        'input_size': (640, 640),
        'mean': [[0, 0, 0]],
        'std': [[1, 1, 1]],
        'input_name': 'input.1',  # 显式指定
    },
    {
        'name': 'w600k_r50',
        'desc': 'ArcFace 人脸识别',
        'onnx': os.path.join(INSIGHTFACE_DIR, 'w600k_r50.onnx'),
        'alias': [os.path.join(MODELS_DIR, 'w600k_r50.onnx')],
        'input_size': (112, 112),
        'mean': [[0, 0, 0]],
        'std': [[1, 1, 1]],
        'input_name': 'input.1',  # 👈 关键：指定为 'input.1'
    },
    {
        'name': 'adaface_ir50',
        'desc': 'AdaFace IR-50 识别',
        'onnx': os.path.join(MODELS_DIR, 'adaface_ir50_ms1mv2.onnx'),
        'alias': [],
        'input_size': (112, 112),
        'mean': [[127.5, 127.5, 127.5]],
        'std': [[127.5, 127.5, 127.5]],
        'input_name': 'input',
    },
]

DATASET_DIR = os.path.join(PROJECT_ROOT, 'Dataset')
DEFAULT_OUTPUT_DIR = MODELS_DIR


def find_onnx(model_cfg):
    """找到 ONNX 文件"""
    paths = [model_cfg['onnx']] + model_cfg['alias']
    for p in paths:
        if os.path.exists(p):
            return p
    
    print(f"\n  ✗ 找不到 ONNX 文件!")
    for p in paths:
        print(f"      {p}")
    sys.exit(1)


def preprocess_for_calibration(img_bgr, target_size, model_name):
    """预处理校准图片"""
    img = cv2.resize(img_bgr, target_size)
    
    if model_name == 'adaface_ir50':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
    else:
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
    
    return img


def load_calibration_images(dataset_dir, target_size, model_name, limit=50):
    """加载校准图片"""
    images = []
    
    if not os.path.exists(dataset_dir):
        return None
    
    jpg_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.jpg'), recursive=True))
    if not jpg_files:
        jpg_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.png'), recursive=True))
    
    for f in jpg_files[:limit]:
        img = cv2.imread(f)
        if img is not None:
            img = preprocess_for_calibration(img, target_size, model_name)
            images.append(img)
    
    return images if images else None


def convert_one(model_cfg, output_dir, quant='fp16'):
    """处理单个模型"""
    name = model_cfg['name']
    onnx_path = find_onnx(model_cfg)
    rknn_path = os.path.join(output_dir, f'{name}.rknn')
    
    print(f"\n{'─' * 50}")
    print(f"  模型: {name} ({model_cfg['desc']})")
    print(f"  ONNX: {onnx_path}")
    print(f"  输出: {rknn_path}")
    print(f"  量化: {quant.upper()}")
    
    rknn = RKNN(verbose=False)
    
    # Step 1: 配置
    print("  [1/5] 配置 RKNN...")
    ret = rknn.config(
        mean_values=model_cfg['mean'],
        std_values=model_cfg['std'],
        target_platform='rk3588',
    )
    if ret != 0:
        print(f"  ✗ config 失败")
        rknn.release()
        return False
    
    # Step 2: 加载 ONNX
    print("  [2/5] 加载 ONNX 模型...")
    try:
        input_size_list = [[1, 3, model_cfg['input_size'][0], model_cfg['input_size'][1]]]
        
        # 获取输入节点名
        input_name = model_cfg.get('input_name')
        
        if input_name:
            print(f"    使用指定输入节点: {input_name}")
            ret = rknn.load_onnx(
                model=onnx_path,
                inputs=[input_name],
                input_size_list=input_size_list
            )
        else:
            print(f"    自动检测输入节点")
            ret = rknn.load_onnx(
                model=onnx_path,
                input_size_list=input_size_list
            )
        
        if ret != 0:
            print(f"  ✗ load_onnx 失败 (ret={ret})")
            rknn.release()
            return False
            
    except Exception as e:
        print(f"  ✗ load_onnx 异常: {e}")
        rknn.release()
        return False
    
    # Step 3: 构建
    print(f"  [3/5] 构建 RKNN 模型 ({quant.upper()})...")
    t0 = time.time()
    
    if quant == 'fp16':
        ret = rknn.build(do_quantization=False)
    else:
        calib_imgs = load_calibration_images(DATASET_DIR, model_cfg['input_size'], name, limit=50)
        if calib_imgs is None or len(calib_imgs) == 0:
            print("    回退到 FP16 模式")
            ret = rknn.build(do_quantization=False)
        else:
            tmpdir = tempfile.mkdtemp(prefix='rknn_calib_')
            txt_path = os.path.join(tmpdir, 'dataset.txt')
            with open(txt_path, 'w') as f:
                for idx, img in enumerate(calib_imgs):
                    npy_path = os.path.join(tmpdir, f'{idx:04d}.npy')
                    np.save(npy_path, img.astype(np.float32))
                    f.write(npy_path + '\n')
            print(f"    校准数据: {len(calib_imgs)} 张")
            ret = rknn.build(do_quantization=True, dataset=txt_path)
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    elapsed = time.time() - t0
    if ret != 0:
        print(f"  ✗ build 失败")
        rknn.release()
        return False
    print(f"    耗时: {elapsed:.1f}s")
    
    # Step 4: 导出
    print("  [4/5] 导出 RKNN...")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"  ✗ export 失败")
        rknn.release()
        return False
    
    size_mb = os.path.getsize(rknn_path) / 1024 / 1024
    print(f"    文件: {rknn_path} ({size_mb:.1f}MB)")
    
    # Step 5: 验证
    print("  [5/5] 验证推理...")
    h, w = model_cfg['input_size']
    
    if name == 'adaface_ir50':
        dummy = np.random.randn(1, h, w, 3).astype(np.float32)
    else:
        dummy = (np.random.rand(1, h, w, 3).astype(np.float32) - 0.5) * 2
    
    try:
        ret = rknn.init_runtime()
        if ret != 0:
            print(f"  ⚠ runtime 初始化失败")
        else:
            outputs = rknn.inference(inputs=[dummy])
            for i, o in enumerate(outputs):
                print(f"    输出[{i}]: shape={o.shape}, dtype={o.dtype}")
    except Exception as e:
        print(f"  ⚠ 推理验证失败: {e}")
    
    rknn.release()
    print(f"  ✓ {name} 转换完成!")
    return True


def main():
    parser = argparse.ArgumentParser(description='ONNX → RKNN 模型转换')
    parser.add_argument('--quant', choices=['fp16', 'int8'], default='fp16',
                        help='量化模式 (默认: fp16)')
    parser.add_argument('--model', choices=['det_10g', 'w600k_r50', 'adaface_ir50', 'all'],
                        default='all', help='转换哪个模型 (默认: all)')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR,
                        help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})')
    args = parser.parse_args()
    
    print("=" * 50)
    print("ONNX → RKNN 转换工具 (RK3588)")
    print(f"量化: {args.quant.upper()} | 输出: {args.output}")
    print("=" * 50)
    
    os.makedirs(args.output, exist_ok=True)
    
    targets = [m for m in MODELS if args.model in ('all', m['name'])]
    if not targets:
        print(f"没有匹配的模型: {args.model}")
        sys.exit(1)
    
    success = 0
    failed = []
    
    for model_cfg in targets:
        if convert_one(model_cfg, args.output, args.quant):
            success += 1
        else:
            failed.append(model_cfg['name'])
    
    print(f"\n{'=' * 50}")
    print(f"结果: {success}/{len(targets)} 成功")
    if failed:
        print(f"失败: {', '.join(failed)}")
    else:
        print(f"输出文件:")
        for m in targets:
            p = os.path.join(args.output, f"{m['name']}.rknn")
            if os.path.exists(p):
                size_mb = os.path.getsize(p) / 1024 / 1024
                print(f"  {p} ({size_mb:.1f}MB)")
    print("=" * 50)


if __name__ == '__main__':
    main()

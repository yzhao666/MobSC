#!/usr/bin/env python3
"""
测试 conformer.onnx 的精度
对比 PyTorch 原始模型与 ONNX 模型的输出差异
"""

import argparse
import os
import sys
import torch
import numpy as np
import onnxruntime as ort
from typing import Tuple

# 添加路径以便导入模块
_this_dir = os.path.dirname(__file__)
_pkg_root = os.path.abspath(os.path.join(_this_dir, ".."))
if _pkg_root not in sys.path:
    sys.path.append(_pkg_root)

from common import set_seed, get_device


def load_pytorch_model(ckpt_path: str, config_path: str, device: torch.device):
    """加载 PyTorch 原始模型"""
    from omegaconf import OmegaConf
    from indextts.gpt.model import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    
    cfg = OmegaConf.load(config_path)
    model = UnifiedVoice(**cfg.gpt)
    load_checkpoint(model, ckpt_path)
    model = model.to(device).eval()
    
    # 提取 Conformer 条件编码器
    conformer = model.conditioning_encoder.to(device).eval()
    return conformer


def create_conformer_wrapper(conformer_model):
    """创建与 ONNX 导出一致的 PyTorch 包装器"""
    class ConformerWrapper(torch.nn.Module):
        def __init__(self, conformer):
            super().__init__()
            self.conformer = conformer

        def forward(self, speech_conditioning_input: torch.Tensor, speech_conditioning_lens: torch.Tensor):
            return self.conformer(speech_conditioning_input, speech_conditioning_lens)
    
    return ConformerWrapper(conformer_model)


def test_accuracy(
    pytorch_model, 
    onnx_path: str, 
    device: torch.device, 
    num_tests: int = 10,
    mel_dim: int = 100,  # 与导出脚本保持一致
    max_seq_len: int = 200
):
    """测试 PyTorch 与 ONNX 模型的精度差异"""
    
    # 加载 ONNX 模型
    onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    print(f">> 开始精度测试，共 {num_tests} 轮...")
    
    max_diff = 0.0
    mean_diff = 0.0
    all_diffs = []
    
    for i in range(num_tests):
        # 生成随机输入 - 注意格式：(batch, time, n_mels)
        seq_len = torch.randint(10, max_seq_len + 1, (1,)).item()
        speech_conditioning_input = torch.randn(1, seq_len, mel_dim, device=device, dtype=torch.float32)
        speech_conditioning_lens = torch.tensor([seq_len], device=device, dtype=torch.long)
        
        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = pytorch_model(speech_conditioning_input, speech_conditioning_lens)
        
        # ONNX 推理
        onnx_inputs = {
            "speech_conditioning_input": speech_conditioning_input.cpu().numpy(),
            "cond_mel_lengths": speech_conditioning_lens.cpu().numpy()
        }
        onnx_outputs = onnx_session.run(None, onnx_inputs)
        onnx_cond_features = onnx_outputs[0]  # cond_features
        onnx_mask = onnx_outputs[1]  # mask
        
        # 计算差异 - 只比较 cond_features
        pytorch_cond_features = pytorch_output[0].cpu().numpy()  # 取第一个输出
        diff = np.abs(pytorch_cond_features - onnx_cond_features).max()
        mean_diff_batch = np.abs(pytorch_cond_features - onnx_cond_features).mean()
        
        max_diff = max(max_diff, diff)
        mean_diff += mean_diff_batch
        all_diffs.append(diff)
        
        print(f"  测试 {i+1:2d}: 最大差异={diff:.2e}, 平均差异={mean_diff_batch:.2e}, 序列长度={seq_len}")
    
    mean_diff /= num_tests
    
    print(f"\n>> 精度测试结果:")
    print(f"  最大差异: {max_diff:.2e}")
    print(f"  平均差异: {mean_diff:.2e}")
    print(f"  差异范围: [{min(all_diffs):.2e}, {max(all_diffs):.2e}]")
    
    # 判断精度是否可接受
    if max_diff < 1e-5:
        print(f"  ✅ 精度优秀 (最大差异 < 1e-5)")
    elif max_diff < 1e-4:
        print(f"  ✅ 精度良好 (最大差异 < 1e-4)")
    elif max_diff < 1e-3:
        print(f"  ⚠️  精度一般 (最大差异 < 1e-3)")
    else:
        print(f"  ❌ 精度较差 (最大差异 >= 1e-3)")
    
    return max_diff, mean_diff


def test_different_sequence_lengths(pytorch_model, onnx_path: str, device: torch.device, mel_dim: int = 100):
    """测试不同序列长度的精度"""
    print(f"\n>> 测试不同序列长度的精度...")
    
    onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    test_lengths = [10, 50, 100, 150, 200, 300]
    
    for seq_len in test_lengths:
        if seq_len > 200:  # 根据模型限制调整
            continue
            
        # 生成输入 - 注意格式：(batch, time, n_mels)
        speech_conditioning_input = torch.randn(1, seq_len, mel_dim, device=device, dtype=torch.float32)
        speech_conditioning_lens = torch.tensor([seq_len], device=device, dtype=torch.long)
        
        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = pytorch_model(speech_conditioning_input, speech_conditioning_lens)
        
        # ONNX 推理
        onnx_inputs = {
            "speech_conditioning_input": speech_conditioning_input.cpu().numpy(),
            "cond_mel_lengths": speech_conditioning_lens.cpu().numpy()
        }
        onnx_outputs = onnx_session.run(None, onnx_inputs)
        onnx_cond_features = onnx_outputs[0]  # cond_features
        
        # 计算差异 - 只比较 cond_features
        pytorch_cond_features = pytorch_output[0].cpu().numpy()
        diff = np.abs(pytorch_cond_features - onnx_cond_features).max()
        print(f"  序列长度 {seq_len:3d}: 最大差异 = {diff:.2e}")


def test_output_distribution(pytorch_model, onnx_path: str, device: torch.device, mel_dim: int = 100):
    """测试输出分布的一致性"""
    print(f"\n>> 测试输出分布一致性...")
    
    onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # 生成测试输入
    seq_len = 100
    speech_conditioning_input = torch.randn(1, mel_dim, seq_len, device=device, dtype=torch.float32)
    speech_conditioning_lens = torch.tensor([seq_len], device=device, dtype=torch.long)
    
    # PyTorch 推理
    with torch.no_grad():
        pytorch_output = pytorch_model(speech_conditioning_input, speech_conditioning_lens)
    
    # ONNX 推理
    onnx_inputs = {
        "speech_conditioning_input": speech_conditioning_input.cpu().numpy(),
        "cond_mel_lengths": speech_conditioning_lens.cpu().numpy()
    }
    onnx_outputs = onnx_session.run(None, onnx_inputs)
    
    # 计算统计信息
    pytorch_cond_features = pytorch_output[0].cpu().numpy()
    onnx_cond_features = onnx_outputs[0]
    
    print(f"  PyTorch 输出统计:")
    print(f"    形状: {pytorch_cond_features.shape}")
    print(f"    均值: {pytorch_cond_features.mean():.6f}")
    print(f"    标准差: {pytorch_cond_features.std():.6f}")
    print(f"    最小值: {pytorch_cond_features.min():.6f}")
    print(f"    最大值: {pytorch_cond_features.max():.6f}")
    
    print(f"  ONNX 输出统计:")
    print(f"    形状: {onnx_cond_features.shape}")
    print(f"    均值: {onnx_cond_features.mean():.6f}")
    print(f"    标准差: {onnx_cond_features.std():.6f}")
    print(f"    最小值: {onnx_cond_features.min():.6f}")
    print(f"    最大值: {onnx_cond_features.max():.6f}")
    
    # 计算分布差异
    mean_diff = abs(pytorch_cond_features.mean() - onnx_cond_features.mean())
    std_diff = abs(pytorch_cond_features.std() - onnx_cond_features.std())
    
    print(f"  分布差异:")
    print(f"    均值差异: {mean_diff:.2e}")
    print(f"    标准差差异: {std_diff:.2e}")


def test_conditioning_quality(pytorch_model, onnx_path: str, device: torch.device, mel_dim: int = 100):
    """测试条件编码的质量（检查输出是否合理）"""
    print(f"\n>> 测试条件编码质量...")
    
    onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # 测试不同长度的输入
    test_cases = [
        (50, "短序列"),
        (100, "中等序列"),
        (150, "长序列")
    ]
    
    for seq_len, desc in test_cases:
        # 生成输入 - 注意格式：(batch, time, n_mels)
        speech_conditioning_input = torch.randn(1, seq_len, mel_dim, device=device, dtype=torch.float32)
        speech_conditioning_lens = torch.tensor([seq_len], device=device, dtype=torch.long)
        
        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = pytorch_model(speech_conditioning_input, speech_conditioning_lens)
        
        # ONNX 推理
        onnx_inputs = {
            "speech_conditioning_input": speech_conditioning_input.cpu().numpy(),
            "cond_mel_lengths": speech_conditioning_lens.cpu().numpy()
        }
        onnx_outputs = onnx_session.run(None, onnx_inputs)
        
        # 检查输出形状和范围
        print(f"  {desc} (长度 {seq_len}):")
        pytorch_cond_features = pytorch_output[0].cpu().numpy()
        onnx_cond_features = onnx_outputs[0]
        print(f"    PyTorch 输出形状: {pytorch_cond_features.shape}")
        print(f"    ONNX 输出形状: {onnx_cond_features.shape}")
        print(f"    PyTorch 输出范围: [{pytorch_cond_features.min():.3f}, {pytorch_cond_features.max():.3f}]")
        print(f"    ONNX 输出范围: [{onnx_cond_features.min():.3f}, {onnx_cond_features.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description="测试 conformer.onnx 精度")
    parser.add_argument("--ckpt", type=str, default="checkpoints/gpt.pth", help="PyTorch 模型路径")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml", help="配置文件路径")
    parser.add_argument("--onnx", type=str, default="checkpoints_onnx/conformer.onnx", help="ONNX 模型路径")
    parser.add_argument("--num_tests", type=int, default=20, help="测试轮数")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    print(f">> 加载 PyTorch 模型: {args.ckpt}")
    conformer_model = load_pytorch_model(args.ckpt, args.config, device)
    
    print(f">> 创建 PyTorch 包装器")
    pytorch_wrapper = create_conformer_wrapper(conformer_model).to(device).eval()
    
    print(f">> 测试 ONNX 模型: {args.onnx}")
    
    # 基本精度测试
    max_diff, mean_diff = test_accuracy(
        pytorch_wrapper, 
        args.onnx, 
        device, 
        num_tests=args.num_tests,
        mel_dim=100  # 与导出脚本保持一致
    )
    
    # 不同序列长度测试
    test_different_sequence_lengths(pytorch_wrapper, args.onnx, device, 100)
    
    # 输出分布测试
    test_output_distribution(pytorch_wrapper, args.onnx, device, 100)
    
    # 条件编码质量测试
    test_conditioning_quality(pytorch_wrapper, args.onnx, device, 100)
    
    print(f"\n>> 测试完成！")


if __name__ == "__main__":
    main()

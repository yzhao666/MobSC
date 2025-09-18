#!/usr/bin/env python3
"""
测试 gpt_core.onnx 的精度
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
    
    return model.gpt, model.model_dim


def create_gpt_core_wrapper(gpt_model):
    """创建与 ONNX 导出一致的 PyTorch 包装器"""
    class GPTCoreWrapper(torch.nn.Module):
        def __init__(self, gpt_module):
            super().__init__()
            self.gpt = gpt_module

        def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor = None):
            out = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
            return out.last_hidden_state
    
    return GPTCoreWrapper(gpt_model)


def test_accuracy(
    pytorch_model, 
    onnx_path: str, 
    device: torch.device, 
    num_tests: int = 10,
    model_dim: int = 1024,
    max_seq_len: int = 64
):
    """测试 PyTorch 与 ONNX 模型的精度差异"""
    
    # 加载 ONNX 模型
    onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    print(f">> 开始精度测试，共 {num_tests} 轮...")
    
    max_diff = 0.0
    mean_diff = 0.0
    all_diffs = []
    
    for i in range(num_tests):
        # 生成随机输入
        seq_len = torch.randint(1, max_seq_len + 1, (1,)).item()
        inputs_embeds = torch.randn(1, seq_len, model_dim, device=device, dtype=torch.float32)
        
        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = pytorch_model(inputs_embeds)
        
        # ONNX 推理
        onnx_inputs = {
            "inputs_embeds": inputs_embeds.cpu().numpy()
        }
        onnx_output = onnx_session.run(None, onnx_inputs)[0]
        
        # 计算差异
        pytorch_np = pytorch_output.cpu().numpy()
        diff = np.abs(pytorch_np - onnx_output).max()
        mean_diff_batch = np.abs(pytorch_np - onnx_output).mean()
        
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


def test_different_sequence_lengths(pytorch_model, onnx_path: str, device: torch.device, model_dim: int):
    """测试不同序列长度的精度"""
    print(f"\n>> 测试不同序列长度的精度...")
    
    onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    test_lengths = [1, 8, 16, 32, 64, 128]
    
    for seq_len in test_lengths:
        if seq_len > 64:  # 根据模型限制调整
            continue
            
        # 生成输入
        inputs_embeds = torch.randn(1, seq_len, model_dim, device=device, dtype=torch.float32)
        
        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = pytorch_model(inputs_embeds)
        
        # ONNX 推理
        onnx_inputs = {
            "inputs_embeds": inputs_embeds.cpu().numpy()
        }
        onnx_output = onnx_session.run(None, onnx_inputs)[0]
        
        # 计算差异
        diff = np.abs(pytorch_output.cpu().numpy() - onnx_output).max()
        print(f"  序列长度 {seq_len:3d}: 最大差异 = {diff:.2e}")


def test_attention_patterns(pytorch_model, onnx_path: str, device: torch.device, model_dim: int):
    """测试不同注意力模式（causal mask）的精度"""
    print(f"\n>> 测试注意力模式精度...")
    
    onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # 测试不同序列长度下的 causal attention
    test_lengths = [8, 16, 32]
    
    for seq_len in test_lengths:
        # 生成输入
        inputs_embeds = torch.randn(1, seq_len, model_dim, device=device, dtype=torch.float32)
        
        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = pytorch_model(inputs_embeds)
        
        # ONNX 推理
        onnx_inputs = {
            "inputs_embeds": inputs_embeds.cpu().numpy()
        }
        onnx_output = onnx_session.run(None, onnx_inputs)[0]
        
        # 计算差异
        diff = np.abs(pytorch_output.cpu().numpy() - onnx_output).max()
        print(f"  Causal attention (长度 {seq_len:2d}): 最大差异 = {diff:.2e}")


def main():
    parser = argparse.ArgumentParser(description="测试 gpt_core.onnx 精度")
    parser.add_argument("--ckpt", type=str, default="checkpoints/gpt.pth", help="PyTorch 模型路径")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml", help="配置文件路径")
    parser.add_argument("--onnx", type=str, default="checkpoints_onnx/gpt_core.onnx", help="ONNX 模型路径")
    parser.add_argument("--num_tests", type=int, default=20, help="测试轮数")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    print(f">> 加载 PyTorch 模型: {args.ckpt}")
    gpt_model, model_dim = load_pytorch_model(args.ckpt, args.config, device)
    
    print(f">> 创建 PyTorch 包装器")
    pytorch_wrapper = create_gpt_core_wrapper(gpt_model).to(device).eval()
    
    print(f">> 测试 ONNX 模型: {args.onnx}")
    
    # 基本精度测试
    max_diff, mean_diff = test_accuracy(
        pytorch_wrapper, 
        args.onnx, 
        device, 
        num_tests=args.num_tests,
        model_dim=model_dim
    )
    
    # 不同序列长度测试
    test_different_sequence_lengths(pytorch_wrapper, args.onnx, device, model_dim)
    
    # 注意力模式测试
    test_attention_patterns(pytorch_wrapper, args.onnx, device, model_dim)
    
    print(f"\n>> 测试完成！")


if __name__ == "__main__":
    main()

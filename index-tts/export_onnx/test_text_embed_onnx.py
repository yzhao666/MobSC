#!/usr/bin/env python3
"""
测试 text_embed.onnx 的精度
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
    
    return model.text_embedding, model.text_pos_embedding, model.model_dim


def create_text_embedding_wrapper(text_embedding, text_pos_embedding):
    """创建与 ONNX 导出一致的 PyTorch 包装器"""
    class TextEmbeddingWrapper(torch.nn.Module):
        def __init__(self, token_emb, pos_emb):
            super().__init__()
            self.token_emb = token_emb
            self.pos_emb = pos_emb

        def forward(self, text_tokens: torch.Tensor, pos_indices: torch.Tensor):
            tok = self.token_emb(text_tokens)
            pos = self.pos_emb.emb(pos_indices)
            return tok + pos
    
    return TextEmbeddingWrapper(text_embedding, text_pos_embedding)


def test_accuracy(
    pytorch_model, 
    onnx_path: str, 
    device: torch.device, 
    num_tests: int = 10,
    vocab_size: int = 256,
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
        text_tokens = torch.randint(0, vocab_size, (1, seq_len), device=device, dtype=torch.long)
        pos_indices = torch.arange(0, seq_len, device=device, dtype=torch.long).unsqueeze(0)
        
        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = pytorch_model(text_tokens, pos_indices)
        
        # ONNX 推理
        onnx_inputs = {
            "text_tokens": text_tokens.cpu().numpy(),
            "pos_indices": pos_indices.cpu().numpy()
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


def test_different_sequence_lengths(pytorch_model, onnx_path: str, device: torch.device):
    """测试不同序列长度的精度"""
    print(f"\n>> 测试不同序列长度的精度...")
    
    onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    vocab_size = 256
    
    test_lengths = [1, 8, 16, 32, 64, 128]
    
    for seq_len in test_lengths:
        if seq_len > 64:  # 根据模型限制调整
            continue
            
        # 生成输入
        text_tokens = torch.randint(0, vocab_size, (1, seq_len), device=device, dtype=torch.long)
        pos_indices = torch.arange(0, seq_len, device=device, dtype=torch.long).unsqueeze(0)
        
        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = pytorch_model(text_tokens, pos_indices)
        
        # ONNX 推理
        onnx_inputs = {
            "text_tokens": text_tokens.cpu().numpy(),
            "pos_indices": pos_indices.cpu().numpy()
        }
        onnx_output = onnx_session.run(None, onnx_inputs)[0]
        
        # 计算差异
        diff = np.abs(pytorch_output.cpu().numpy() - onnx_output).max()
        print(f"  序列长度 {seq_len:3d}: 最大差异 = {diff:.2e}")


def main():
    parser = argparse.ArgumentParser(description="测试 text_embed.onnx 精度")
    parser.add_argument("--ckpt", type=str, default="checkpoints/gpt.pth", help="PyTorch 模型路径")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml", help="配置文件路径")
    parser.add_argument("--onnx", type=str, default="checkpoints_onnx/text_embed.onnx", help="ONNX 模型路径")
    parser.add_argument("--num_tests", type=int, default=20, help="测试轮数")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    print(f">> 加载 PyTorch 模型: {args.ckpt}")
    text_embedding, text_pos_embedding, model_dim = load_pytorch_model(args.ckpt, args.config, device)
    
    print(f">> 创建 PyTorch 包装器")
    pytorch_wrapper = create_text_embedding_wrapper(text_embedding, text_pos_embedding).to(device).eval()
    
    print(f">> 测试 ONNX 模型: {args.onnx}")
    
    # 基本精度测试
    max_diff, mean_diff = test_accuracy(
        pytorch_wrapper, 
        args.onnx, 
        device, 
        num_tests=args.num_tests,
        vocab_size=text_embedding.num_embeddings
    )
    
    # 不同序列长度测试
    test_different_sequence_lengths(pytorch_wrapper, args.onnx, device)
    
    print(f"\n>> 测试完成！")


if __name__ == "__main__":
    main()

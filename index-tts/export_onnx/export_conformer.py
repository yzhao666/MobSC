import argparse
import os
from typing import Tuple

import torch
import numpy as np
import onnxruntime as ort

try:
    from .common import set_seed, get_device, verify_onnx
except Exception:
    import sys
    _this_dir = os.path.dirname(__file__)
    sys.path.append(_this_dir)
    _pkg_root = os.path.abspath(os.path.join(_this_dir, ".."))
    if _pkg_root not in sys.path:
        sys.path.append(_pkg_root)
    from common import set_seed, get_device, verify_onnx


def load_conformer_model(ckpt_path: str, config_path: str, device: torch.device) -> torch.nn.Module:
    """从 GPT 模型中提取 Conformer 条件编码器"""
    from omegaconf import OmegaConf
    from indextts.gpt.model import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    
    # 加载配置
    cfg = OmegaConf.load(config_path)
    
    # 创建完整的 GPT 模型
    gpt_model = UnifiedVoice(**cfg.gpt)
    
    # 加载权重
    load_checkpoint(gpt_model, ckpt_path)
    
    # 提取 Conformer 条件编码器
    conformer_model = gpt_model.conditioning_encoder
    
    # 设置为评估模式并移动到设备
    conformer_model = conformer_model.to(device)
    conformer_model.eval()
    
    print(f">> Conformer 条件编码器提取完成: {ckpt_path}")
    return conformer_model


def build_dummy_inputs(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """构造 Conformer 模型的虚拟输入"""
    # 根据 IndexTTS Conformer 的输入格式构造
    # speech_conditioning_input: (batch, n_mels, frames) -> (batch, frames, n_mels)
    # 根据配置文件：n_mels=100, 使用合理的时间维度
    speech_conditioning_input = torch.randn(1, 200, 100, device=device).float()  # (batch, time, n_mels)
    
    # cond_mel_lengths: (batch,)
    cond_mel_lengths = torch.tensor([200], device=device).long()
    
    return speech_conditioning_input, cond_mel_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/gpt.pth")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml")
    parser.add_argument("--out", type=str, default="checkpoints_onnx/conformer.onnx",
                       help="Output ONNX model path (default: checkpoints_onnx/conformer.onnx)")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--no-test", action="store_true", help="导出后跳过快速精度测试")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = load_conformer_model(args.ckpt, args.config, device)
    model.eval()

    speech_conditioning_input, cond_mel_lengths = build_dummy_inputs(device)

    input_names = ["speech_conditioning_input", "cond_mel_lengths"]
    output_names = ["cond_features", "mask"]
    dynamic_axes = {
        "speech_conditioning_input": {0: "batch", 1: "time"},
        "cond_mel_lengths": {0: "batch"},
        "cond_features": {0: "batch", 1: "time"},
        "mask": {0: "batch", 2: "time"},
    }

    torch.onnx.export(
        model,
        (speech_conditioning_input, cond_mel_lengths),
        args.out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    verify_onnx(args.out)
    print(f"[OK] Exported Conformer ONNX to {args.out}")

    if not args.no_test:
        try:
            sess = ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
            onnx_outs = sess.run(None, {
                "speech_conditioning_input": speech_conditioning_input.detach().cpu().numpy(),
                "cond_mel_lengths": cond_mel_lengths.detach().cpu().numpy(),
            })
            onnx_feat = onnx_outs[0]
            with torch.no_grad():
                pt_out = model(speech_conditioning_input, cond_mel_lengths)
                pt_feat = pt_out[0].detach().cpu().numpy() if isinstance(pt_out, (list, tuple)) else pt_out.detach().cpu().numpy()
            max_diff = float(np.max(np.abs(pt_feat - onnx_feat)))
            mean_diff = float(np.mean(np.abs(pt_feat - onnx_feat)))
            print(f"[TEST] conformer.onnx 精度: max={max_diff:.2e}, mean={mean_diff:.2e}")
        except Exception as e:
            print(f"[TEST] conformer.onnx 精度测试失败: {e}")


if __name__ == "__main__":
    main()



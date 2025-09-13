import argparse
import os
from typing import Tuple

import torch

try:
    # 作为包运行: python -m export_onnx.export_gpt
    from .common import set_seed, get_device, verify_onnx
except Exception:  # 直接在仓库根目录运行: python index-tts/export_onnx/export_gpt.py
    import sys
    _this_dir = os.path.dirname(__file__)
    sys.path.append(_this_dir)
    # 让 "indextts" 可被发现（父目录为 index-tts）
    _pkg_root = os.path.abspath(os.path.join(_this_dir, ".."))
    if _pkg_root not in sys.path:
        sys.path.append(_pkg_root)
    from common import set_seed, get_device, verify_onnx


def load_gpt_model(ckpt_path: str, config_path: str, device: torch.device) -> torch.nn.Module:
    """加载 IndexTTS GPT 模型"""
    from omegaconf import OmegaConf
    from indextts.gpt.model import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    
    # 加载配置
    cfg = OmegaConf.load(config_path)
    
    # 创建模型
    model = UnifiedVoice(**cfg.gpt)
    
    # 加载权重
    load_checkpoint(model, ckpt_path)
    
    # 设置为评估模式并移动到设备
    model = model.to(device)
    model.eval()
    
    # 初始化 GPT2 配置（如果需要）
    if hasattr(model, 'post_init_gpt2_config'):
        model.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)
    
    print(f">> GPT 模型加载完成: {ckpt_path}")
    return model


def build_dummy_inputs(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造 GPT 模型的虚拟输入"""
    # 根据 IndexTTS GPT 模型的输入格式构造
    # speech_conditioning_mel: (1, n_mels, frames)
    # 根据配置文件：n_mels=100, hop_length=256
    # 使用更合理的时间维度，避免子采样层的维度不匹配
    speech_conditioning_mel = torch.randn(1, 100, 512, device=device).float()  # (batch, n_mels, time_frames)
    
    # text_tokens: (1, seq_len)
    text_tokens = torch.randint(1, 1000, (1, 128), device=device).long()
    
    # cond_mel_lengths: (1,)
    cond_mel_lengths = torch.tensor([512], device=device).long()  # 匹配时间维度
    
    return speech_conditioning_mel, text_tokens, cond_mel_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/gpt.pth")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml")
    parser.add_argument("--out", type=str, default="checkpoints_onnx/gpt.onnx", 
                       help="Output ONNX model path (default: checkpoints_onnx/gpt.onnx)")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version (default: 18 for ONNX 1.19.0+)")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"], default="fp32", 
                       help="Export precision: fp32, fp16, or int8")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # 根据精度调整输出文件名
    if args.precision != "fp32":
        base_name = os.path.splitext(args.out)[0]
        ext = os.path.splitext(args.out)[1]
        args.out = f"{base_name}_{args.precision}{ext}"
        print(f">> 输出文件调整为: {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = load_gpt_model(args.ckpt, args.config, device)
    
    # 根据精度要求转换模型
    if args.precision == "fp16":
        model = model.half()
        print(">> 模型转换为 FP16 精度")
    elif args.precision == "int8":
        # INT8 量化需要特殊处理，这里先提示
        print(">> 警告: INT8 量化需要额外的量化步骤，当前导出为 FP32")
        print(">> 建议使用 onnxruntime 的量化工具进行后处理")
    
    speech_conditioning_mel, text_tokens, cond_mel_lengths = build_dummy_inputs(device)
    
    # 根据精度调整输入数据类型
    if args.precision == "fp16":
        speech_conditioning_mel = speech_conditioning_mel.half()

    input_names = ["speech_conditioning_mel", "text_tokens", "cond_mel_lengths"]
    output_names = ["codes"]
    dynamic_axes = {
        "speech_conditioning_mel": {0: "batch", 2: "time"},
        "text_tokens": {0: "batch", 1: "time"},
        "cond_mel_lengths": {0: "batch"},
        "codes": {0: "batch", 1: "time"},
    }

    # 注意：这里需要根据实际的 GPT 模型 forward 方法来调整
    # 可能需要创建一个包装器来简化导出
    class GPTWrapper(torch.nn.Module):
        def __init__(self, gpt_model):
            super().__init__()
            self.gpt = gpt_model
        
        def forward(self, speech_conditioning_mel, text_tokens, cond_mel_lengths):
            # 调用 inference_speech 方法
            return self.gpt.inference_speech(
                speech_conditioning_mel, 
                text_tokens, 
                cond_mel_lengths=cond_mel_lengths,
                max_generate_length=600
            )
    
    wrapped_model = GPTWrapper(model)

    torch.onnx.export(
        wrapped_model,
        (speech_conditioning_mel, text_tokens, cond_mel_lengths),
        args.out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    verify_onnx(args.out)
    print(f"[OK] Exported GPT ONNX to {args.out}")
    
    # 如果是 INT8 精度，进行量化
    if args.precision == "int8":
        quantized_path = args.out.replace(".onnx", "_quantized.onnx")
        quantize_onnx_model(args.out, quantized_path)
        print(f"[OK] Quantized model saved to {quantized_path}")


def quantize_onnx_model(input_path: str, output_path: str):
    """使用 ONNXRuntime 量化 ONNX 模型"""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        print(">> 开始 INT8 量化...")
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=['MatMul', 'Gemm', 'Conv', 'Add', 'Mul']
        )
        print(">> INT8 量化完成")
        
        # 验证量化后的模型
        verify_onnx(output_path)
        
    except ImportError:
        print(">> 警告: 无法导入 onnxruntime.quantization，跳过量化")
        print(">> 请安装: pip install onnxruntime[quantization]")
    except Exception as e:
        print(f">> 量化失败: {e}")
        print(">> 请检查模型是否支持量化")


if __name__ == "__main__":
    main()



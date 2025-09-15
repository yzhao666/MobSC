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


def build_dummy_inputs(device: torch.device, max_mel_frames: int = 256, max_text_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造 GPT 模型的虚拟输入 - 使用可配置的尺寸减少内存消耗"""
    # 使用可配置的输入尺寸来减少内存消耗和导出时间
    # speech_conditioning_mel: (1, n_mels, frames)
    speech_conditioning_mel = torch.randn(1, 100, max_mel_frames, device=device).float()
    
    # text_tokens: (1, seq_len) - 可配置序列长度
    text_tokens = torch.randint(1, 1000, (1, max_text_len), device=device).long()
    
    # cond_mel_lengths: (1,)
    cond_mel_lengths = torch.tensor([max_mel_frames], device=device).long()
    
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
    parser.add_argument("--max-mel-frames", type=int, default=256, 
                       help="Maximum mel frames for export (default: 256)")
    parser.add_argument("--max-text-len", type=int, default=64, 
                       help="Maximum text sequence length for export (default: 64)")
    parser.add_argument("--use-torch-export", action="store_true", 
                       help="Use new torch.export instead of legacy ONNX export")
    parser.add_argument("--minimal-export", action="store_true", 
                       help="Use minimal export with smallest possible inputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    
    # 内存优化设置
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 设置内存分配策略
        torch.cuda.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用
        print(f">> GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 设置线程数以减少内存使用
    torch.set_num_threads(1)
    print(f">> 使用CPU线程数: {torch.get_num_threads()}")

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
    
    # 根据minimal-export选项调整输入尺寸
    if args.minimal_export:
        print(">> 使用最小化导出模式...")
        mel_frames = min(128, args.max_mel_frames)  # 进一步减少
        text_len = min(32, args.max_text_len)
    else:
        mel_frames = args.max_mel_frames
        text_len = args.max_text_len
    
    speech_conditioning_mel, text_tokens, cond_mel_lengths = build_dummy_inputs(
        device, mel_frames, text_len
    )
    
    # 根据精度调整输入数据类型
    if args.precision == "fp16":
        speech_conditioning_mel = speech_conditioning_mel.half()

    input_names = ["speech_conditioning_mel", "text_tokens", "cond_mel_lengths"]
    output_names = ["logits"]
    dynamic_axes = {
        "speech_conditioning_mel": {0: "batch", 2: "time"},
        "text_tokens": {0: "batch", 1: "time"},
        "cond_mel_lengths": {0: "batch"},
        "logits": {0: "batch", 1: "time", 2: "vocab"},
    }

    # 创建简化的GPT包装器，避免使用复杂的generate方法
    class GPTWrapper(torch.nn.Module):
        def __init__(self, gpt_model):
            super().__init__()
            self.gpt = gpt_model
        
        def forward(self, speech_conditioning_mel, text_tokens, cond_mel_lengths):
            """简化的forward方法，只进行单步推理而不是完整生成"""
            # 获取条件编码
            conds_latent = self.gpt.get_conditioning(speech_conditioning_mel, cond_mel_lengths)
            
            # 准备GPT输入
            input_ids, inputs_embeds, attention_mask = self.gpt.prepare_gpt_inputs(conds_latent, text_tokens)
            
            # 存储mel embedding
            self.gpt.inference_model.store_mel_emb(inputs_embeds)
            
            # 只进行单步forward，不进行生成
            with torch.no_grad():
                # 获取GPT2InferenceModel的forward输出
                outputs = self.gpt.inference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False
                )
                
                # 返回logits，而不是生成的tokens
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                # 只返回最后一个token的logits，模拟单步生成
                return logits[:, -1:, :]  # (batch, 1, vocab_size)
    
    wrapped_model = GPTWrapper(model)

    print(">> 开始导出ONNX模型...")
    print(f">> 输入形状: speech_conditioning_mel={speech_conditioning_mel.shape}, text_tokens={text_tokens.shape}, cond_mel_lengths={cond_mel_lengths.shape}")
    
    if args.use_torch_export:
        print(">> 使用新的 torch.export 方法...")
        try:
            # 使用新的 torch.export 方法
            exported_program = torch.export.export(
                wrapped_model,
                (speech_conditioning_mel, text_tokens, cond_mel_lengths),
            )
            
            # 转换为ONNX
            from torch.onnx import export as onnx_export
            onnx_export(
                exported_program,
                (speech_conditioning_mel, text_tokens, cond_mel_lengths),
                args.out,
                export_params=True,
                opset_version=args.opset,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        except Exception as e:
            print(f">> torch.export 失败，回退到传统方法: {e}")
            # 回退到传统方法
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
                verbose=False,
                keep_initializers_as_inputs=False,
                training=torch.onnx.TrainingMode.EVAL,
            )
    else:
        # 使用传统导出方法，但添加更多优化
        print(">> 使用传统 torch.onnx.export 方法...")
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
            verbose=False,  # 减少输出信息
            keep_initializers_as_inputs=False,  # 优化模型大小
            training=torch.onnx.TrainingMode.EVAL,  # 确保是推理模式
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



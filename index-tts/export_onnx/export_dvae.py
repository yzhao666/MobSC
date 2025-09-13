import argparse
import os
from typing import Tuple

import torch

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


def load_dvae_model(ckpt_path: str, config_path: str, device: torch.device) -> torch.nn.Module:
    # TODO: 替换为 indextts DVAE/VQ 模块的真实装载逻辑
    raise NotImplementedError("请根据 indextts 实际实现补齐 DVAE/VQ 模块的装载逻辑")


def build_dummy_inputs(device: torch.device) -> Tuple[torch.Tensor]:
    # 示例：来自 GPT 的中间特征或 codes
    latents = torch.randn(1, 256, 200, device=device)  # (B, C, T) 占位
    return (latents,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/dvae.pth")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml")
    parser.add_argument("--out", type=str, default="checkpoints_onnx/dvae.onnx",
                       help="Output ONNX model path (default: checkpoints_onnx/dvae.onnx)")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = load_dvae_model(args.ckpt, args.config, device)
    model.eval()

    (latents,) = build_dummy_inputs(device)

    input_names = ["latents"]
    output_names = ["decoder_features"]
    dynamic_axes = {
        "latents": {0: "batch", 2: "time"},
        "decoder_features": {0: "batch", 2: "time"},
    }

    torch.onnx.export(
        model,
        (latents,),
        args.out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    verify_onnx(args.out)
    print(f"[OK] Exported DVAE ONNX to {args.out}")


if __name__ == "__main__":
    main()



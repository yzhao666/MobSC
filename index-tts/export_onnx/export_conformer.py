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


def load_conformer_model(ckpt_path: str, config_path: str, device: torch.device) -> torch.nn.Module:
    # TODO: 替换为 indextts 条件编码器的真实装载逻辑
    raise NotImplementedError("请根据 indextts 实际实现补齐 Conformer/Condition encoder 的装载逻辑")


def build_dummy_inputs(device: torch.device) -> Tuple[torch.Tensor]:
    # 示例：梅尔谱特征或音频/特征序列等
    feats = torch.randn(1, 80, 200, device=device)  # (B, C, T) 仅示例
    return (feats,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default="checkpoints_onnx/conformer.onnx",
                       help="Output ONNX model path (default: checkpoints_onnx/conformer.onnx)")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = load_conformer_model(args.ckpt, args.config, device)
    model.eval()

    (feats,) = build_dummy_inputs(device)

    input_names = ["feats"]
    output_names = ["cond_features"]
    dynamic_axes = {
        "feats": {0: "batch", 2: "time"},
        "cond_features": {0: "batch", 1: "time"},
    }

    torch.onnx.export(
        model,
        (feats,),
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


if __name__ == "__main__":
    main()



import argparse
import os
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


def load_lm_head(ckpt_path: str, config_path: str, device: torch.device):
    from omegaconf import OmegaConf
    from indextts.gpt.model import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint

    cfg = OmegaConf.load(config_path)
    m = UnifiedVoice(**cfg.gpt)
    load_checkpoint(m, ckpt_path)
    m = m.to(device).eval()
    # 获取 final_norm 和 mel_head 组件
    final_norm = m.final_norm.to(device).eval()
    mel_head = m.mel_head.to(device).eval()
    dim = m.model_dim
    vocab = m.number_mel_codes
    return final_norm, mel_head, dim, vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/gpt.pth")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml")
    parser.add_argument("--out", type=str, default="checkpoints_onnx/lm_head.onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    final_norm, mel_head, dim, vocab = load_lm_head(args.ckpt, args.config, device)

    class LMHeadWrapper(torch.nn.Module):
        def __init__(self, final_norm, mel_head):
            super().__init__()
            self.final_norm = final_norm
            self.mel_head = mel_head

        def forward(self, hidden_states: torch.Tensor):
            normalized_hidden_states = self.final_norm(hidden_states)
            logits = self.mel_head(normalized_hidden_states)
            return logits

    wrapper = LMHeadWrapper(final_norm, mel_head).to(device).eval()

    B, T = 1, 64
    hidden_states = torch.randn(B, T, dim, device=device, dtype=torch.float32)

    input_names = ["hidden_states"]
    output_names = ["logits"]
    dynamic_axes = {
        "hidden_states": {0: "batch", 1: "time"},
        "logits": {0: "batch", 1: "time", 2: "vocab"},
    }

    torch.onnx.export(
        wrapper,
        (hidden_states,),
        args.out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
    )

    verify_onnx(args.out)
    print(f"[OK] Exported LM Head ONNX to {args.out}")


if __name__ == "__main__":
    main()



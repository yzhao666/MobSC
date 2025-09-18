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


def load_gpt_core(ckpt_path: str, config_path: str, device: torch.device):
    from omegaconf import OmegaConf
    from indextts.gpt.model import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint

    cfg = OmegaConf.load(config_path)
    m = UnifiedVoice(**cfg.gpt)
    load_checkpoint(m, ckpt_path)
    m = m.to(device).eval()
    # 提取纯 GPT transformer（不含 lm_head 和 embeddings）
    gpt = m.gpt.to(device).eval()
    model_dim = m.model_dim
    return gpt, model_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/gpt.pth")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml")
    parser.add_argument("--out", type=str, default="checkpoints_onnx/gpt_core.onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    gpt, dim = load_gpt_core(args.ckpt, args.config, device)

    class GPTCoreWrapper(torch.nn.Module):
        def __init__(self, gpt_module):
            super().__init__()
            self.gpt = gpt_module

        def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor = None):
            out = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
            return out.last_hidden_state

    wrapper = GPTCoreWrapper(gpt).to(device).eval()

    B, T = 1, 64
    inputs_embeds = torch.randn(B, T, dim, device=device, dtype=torch.float32)
    # GPT2 期望的 attention mask 格式: [B, 1, 1, T] 或 None
    # 使用 None 让模型自动生成 causal mask
    attn_mask = None

    input_names = ["inputs_embeds"]
    output_names = ["hidden_states"]
    dynamic_axes = {
        "inputs_embeds": {0: "batch", 1: "time"},
        "hidden_states": {0: "batch", 1: "time"},
    }

    torch.onnx.export(
        wrapper,
        (inputs_embeds,),
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
    print(f"[OK] Exported GPT Core ONNX to {args.out}")


if __name__ == "__main__":
    main()



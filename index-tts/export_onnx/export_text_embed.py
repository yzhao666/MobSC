import argparse
import os
from typing import Tuple

import torch

try:
    # 作为包运行: python -m export_onnx.export_text_embed
    from .common import set_seed, get_device, verify_onnx
except Exception:  # 直接在仓库根目录运行
    import sys
    _this_dir = os.path.dirname(__file__)
    sys.path.append(_this_dir)
    _pkg_root = os.path.abspath(os.path.join(_this_dir, ".."))
    if _pkg_root not in sys.path:
        sys.path.append(_pkg_root)
    from common import set_seed, get_device, verify_onnx


def load_model_parts(ckpt_path: str, config_path: str, device: torch.device):
    """加载 UnifiedVoice 并返回 text_embedding 与 text_pos_embedding."""
    from omegaconf import OmegaConf
    from indextts.gpt.model import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint

    cfg = OmegaConf.load(config_path)
    model = UnifiedVoice(**cfg.gpt)
    load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()
    text_embedding = model.text_embedding
    text_pos_embedding = model.text_pos_embedding  # LearnedPositionEmbeddings
    return text_embedding.to(device).eval(), text_pos_embedding.to(device).eval(), model.model_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/gpt.pth")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml")
    parser.add_argument("--out", type=str, default="checkpoints_onnx/text_embed.onnx",
                        help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    text_embedding, text_pos_embedding, model_dim = load_model_parts(args.ckpt, args.config, device)

    class TextEmbeddingWrapper(torch.nn.Module):
        def __init__(self, token_emb, pos_emb):
            super().__init__()
            self.token_emb = token_emb
            self.pos_emb = pos_emb  # has .emb lookup table

        def forward(self, text_tokens: torch.Tensor, pos_indices: torch.Tensor):
            # text_tokens: (B, T) int64; pos_indices: (B, T) int64
            tok = self.token_emb(text_tokens)
            # position embedding: lookup via pos_indices values
            # LearnedPositionEmbeddings stores indices in .emb, so we use embedding-like lookup
            pos = self.pos_emb.emb(pos_indices)
            return tok + pos  # (B, T, D)

    wrapper = TextEmbeddingWrapper(text_embedding, text_pos_embedding).to(device).eval()

    # Dummy inputs
    B, T = 1, 32
    text_tokens = torch.randint(0, text_embedding.num_embeddings, (B, T), device=device, dtype=torch.long)
    pos_indices = torch.arange(0, T, device=device, dtype=torch.long).unsqueeze(0).repeat(B, 1)

    input_names = ["text_tokens", "pos_indices"]
    output_names = ["text_embeddings"]
    dynamic_axes = {
        "text_tokens": {0: "batch", 1: "time"},
        "pos_indices": {0: "batch", 1: "time"},
        "text_embeddings": {0: "batch", 1: "time"},
    }

    torch.onnx.export(
        wrapper,
        (text_tokens, pos_indices),
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
    print(f"[OK] Exported Text Embedding ONNX to {args.out}")


if __name__ == "__main__":
    main()



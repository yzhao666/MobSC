import os
import random
import torch


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_dynamic_axes(batch_dim: bool = True):
    axes = {"input": {}}  # caller will rename inputs/outputs
    if batch_dim:
        axes["input"][0] = "batch"
    return axes


def verify_onnx(path: str) -> None:
    import onnx
    model = onnx.load(path)
    onnx.checker.check_model(model)



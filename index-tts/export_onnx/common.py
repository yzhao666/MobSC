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


def verify_onnx(path: str) -> bool:
    """验证 ONNX 模型并返回验证结果"""
    try:
        import onnx
        model = onnx.load(path)
        onnx.checker.check_model(model)
        print(f">> ONNX 模型验证通过: {path}")
        print(f">> 模型输入: {[input.name for input in model.graph.input]}")
        print(f">> 模型输出: {[output.name for output in model.graph.output]}")
        return True
    except Exception as e:
        print(f">> ONNX 模型验证失败: {path}")
        print(f">> 错误信息: {e}")
        return False



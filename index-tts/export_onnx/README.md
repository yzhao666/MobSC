## index-tts ONNX 导出脚本

本目录提供将 `index-tts` 的各子模型导出为 ONNX 的脚本：GPT、Conformer（条件编码器）、DVAE/VQ、BigVGAN2 生成器，以及一键导出脚本。

### 环境要求
- Python ≥ 3.10
- PyTorch 与与之匹配的 CUDA/CPU 版本
- onnx, onnxruntime（用于基本检查）

### 目录结构
```
export_onnx/
  ├─ common.py              # 通用工具：seed、device、动态轴、检查函数
  ├─ export_gpt.py          # 导出 GPT 样式声学模型
  ├─ export_conformer.py    # 导出条件编码器（例如 Conformer）
  ├─ export_dvae.py         # 导出 DVAE/VQ 等编码器/解码器子图
  ├─ export_bigvgan.py      # 导出 BigVGAN2 生成器
  └─ export_all.py          # 一键导出，调用以上脚本
```

### 使用示例
```bash
# 建议在 index-tts 根目录执行

# 基本导出 (FP32) - 使用默认路径 checkpoints_onnx/
python -m export_onnx.export_gpt --ckpt checkpoints/gpt.pth --config checkpoints/config.yaml
python -m export_onnx.export_conformer --ckpt checkpoints/conformer.pth --config checkpoints/config.yaml
python -m export_onnx.export_dvae --ckpt checkpoints/dvae.pth --config checkpoints/config.yaml
python -m export_onnx.export_bigvgan --ckpt checkpoints/bigvgan_generator.pth --config checkpoints/config.yaml

# 不同精度导出
python -m export_onnx.export_gpt --ckpt checkpoints/gpt.pth --config checkpoints/config.yaml --precision fp16
python -m export_onnx.export_gpt --ckpt checkpoints/gpt.pth --config checkpoints/config.yaml --precision int8

# 指定 ONNX opset 版本 (默认 18，适用于 ONNX 1.19.0+)
python -m export_onnx.export_gpt --ckpt checkpoints/gpt.pth --config checkpoints/config.yaml --opset 18

# 自定义输出路径
python -m export_onnx.export_gpt --ckpt checkpoints/gpt.pth --config checkpoints/config.yaml --out custom_path/gpt.onnx

# 一键导出
python -m export_onnx.export_all --config checkpoints/config.yaml --model_dir checkpoints --out_dir checkpoints_onnx
```

### 默认输出目录
所有导出的 ONNX 模型默认保存在 `checkpoints_onnx/` 目录下：
- `checkpoints_onnx/gpt.onnx` - GPT 模型
- `checkpoints_onnx/conformer.onnx` - Conformer 条件编码器
- `checkpoints_onnx/dvae.onnx` - DVAE 模型
- `checkpoints_onnx/bigvgan.onnx` - BigVGAN 生成器

### 精度说明
- **FP32**: 默认精度，兼容性最好，文件最大
- **FP16**: 半精度，文件大小减半，推理速度提升，需要 GPU 支持
- **INT8**: 量化精度，文件最小，推理最快，但可能影响音质

> 注：脚本中的模型装载函数需按 `indextts` 实际模块与权重字段补齐；当前提供了清晰的 TODO 标记与占位实现接口。



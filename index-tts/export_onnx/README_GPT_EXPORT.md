# GPT模型ONNX导出优化指南

## 问题描述

原始的GPT模型导出脚本在导出ONNX时会出现以下问题：
1. 导出过程卡住，需要1小时以上甚至被系统kill
2. 内存消耗过大，导致OOM错误
3. 使用了复杂的`inference_speech`方法，包含HuggingFace的`generate`逻辑

## 解决方案

### 1. 简化模型包装器

创建了`GPTWrapper`类，避免使用复杂的生成逻辑：
- 只进行单步forward推理，不进行完整生成
- 返回logits而不是生成的tokens
- 移除了beam search、sampling等复杂逻辑

### 2. 内存优化

- 减少输入尺寸：默认mel_frames=256, text_len=64
- 添加最小化导出模式：mel_frames=128, text_len=32
- 限制GPU内存使用：`torch.cuda.set_per_process_memory_fraction(0.8)`
- 设置单线程：`torch.set_num_threads(1)`

### 3. 导出选项优化

- 添加`--minimal-export`选项用于最小化导出
- 添加`--max-mel-frames`和`--max-text-len`参数控制输入尺寸
- 添加`--use-torch-export`选项尝试新的导出方法
- 优化ONNX导出参数：`verbose=False`, `keep_initializers_as_inputs=False`

## 使用方法

### 基本导出

```bash
# 最小化导出（推荐先尝试）
python export_onnx/export_gpt.py --minimal-export

# 小尺寸导出
python export_onnx/export_gpt.py --max-mel-frames 128 --max-text-len 32

# FP16导出
python export_onnx/export_gpt.py --precision fp16 --max-mel-frames 128 --max-text-len 32
```

### 测试脚本

使用测试脚本验证不同配置的导出：

```bash
python export_onnx/test_gpt_export.py --ckpt checkpoints/gpt.pth --config checkpoints/config.yaml
```

### 参数说明

- `--minimal-export`: 使用最小输入尺寸（mel_frames=64, text_len=16）
- `--max-mel-frames`: 最大mel帧数（默认256）
- `--max-text-len`: 最大文本长度（默认64）
- `--precision`: 精度选择（fp32/fp16/int8）
- `--use-torch-export`: 使用新的torch.export方法
- `--opset`: ONNX opset版本（默认18）

## 导出策略建议

### 1. 渐进式导出

```bash
# 第一步：最小化导出测试
python export_onnx/export_gpt.py --minimal-export

# 第二步：如果成功，尝试稍大的尺寸
python export_onnx/export_gpt.py --max-mel-frames 128 --max-text-len 32

# 第三步：如果成功，尝试FP16
python export_onnx/export_gpt.py --precision fp16 --max-mel-frames 128 --max-text-len 32
```

### 2. 内存不足时的处理

如果仍然遇到内存问题：

1. 进一步减少输入尺寸：
   ```bash
   python export_onnx/export_gpt.py --minimal-export --max-mel-frames 64 --max-text-len 16
   ```

2. 使用CPU导出：
   ```bash
   CUDA_VISIBLE_DEVICES="" python export_onnx/export_gpt.py --minimal-export
   ```

3. 尝试新的导出方法：
   ```bash
   python export_onnx/export_gpt.py --minimal-export --use-torch-export
   ```

## 输出说明

导出的ONNX模型包含以下输入输出：

### 输入
- `speech_conditioning_mel`: (batch, n_mels, time) - 语音条件mel谱
- `text_tokens`: (batch, seq_len) - 文本token序列
- `cond_mel_lengths`: (batch,) - 条件mel长度

### 输出
- `logits`: (batch, 1, vocab_size) - 下一个token的logits

## 注意事项

1. **模型功能限制**: 导出的模型只支持单步推理，不支持完整的生成过程
2. **输入尺寸**: 实际使用时需要确保输入尺寸不超过导出时的设置
3. **精度**: FP16可能在某些硬件上不兼容，建议先测试FP32
4. **内存**: 如果仍然遇到内存问题，可以进一步减少输入尺寸

## 故障排除

### 导出卡住
- 使用`--minimal-export`选项
- 检查系统内存使用情况
- 尝试CPU导出

### 内存不足
- 减少`--max-mel-frames`和`--max-text-len`参数
- 使用`--minimal-export`选项
- 关闭其他占用内存的程序

### 精度问题
- 先使用FP32确保功能正常
- FP16可能在某些操作上不兼容
- INT8需要额外的量化步骤

## 后续步骤

1. 验证导出的ONNX模型可以正常加载
2. 使用MNNConvert转换为MNN格式
3. 在Android端集成和测试

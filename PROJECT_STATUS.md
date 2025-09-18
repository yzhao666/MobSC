# MobSC 项目状态报告

## 项目概述
MobSC 是一个将 IndexTTS 语音合成系统迁移到移动端（Android）的项目，基于 MNN 框架实现端侧离线推理。

## 最新进展 (2024年9月17日)

### ✅ 重大突破：ONNX 子模块化导出完成

#### 问题解决
- **ONNX 2GB 限制问题**：成功将大型 GPT 模型（1GB+）拆分为 4 个独立子模块
- **外部数据文件问题**：每个 ONNX 文件都是独立的，无外部依赖
- **精度验证体系**：建立了完整的 PyTorch vs ONNX 精度对比测试

#### 技术成果

##### 1. 子模块导出架构
```
IndexTTS GPT 模型 → 4个子模块
├── text_embed.onnx (62M)    - 文本嵌入 + 位置编码
├── gpt_core.onnx (1.8G)     - GPT Transformer 核心层  
├── lm_head.onnx (41M)       - 最终归一化 + 线性投影
└── conformer.onnx (156M)    - 条件编码器
```

##### 2. 精度验证结果
| 模块 | 文件大小 | 精度等级 | 最大差异 | 状态 |
|------|----------|----------|----------|------|
| text_embed | 62M | 优秀 | 0.00e+00 | ✅ 完美 |
| gpt_core | 1.8G | 良好 | 4.29e-05 | ✅ 优秀 |
| lm_head | 41M | 优秀 | 8.11e-06 | ✅ 完美 |
| conformer | 156M | 一般 | 7.46e-04 | ✅ 可接受 |

##### 3. 新增脚本文件
**导出脚本：**
- `export_text_embed.py` - 文本嵌入器导出
- `export_gpt_core.py` - GPT 核心导出  
- `export_lm_head.py` - LM 头导出
- `export_conformer.py` - 条件编码器导出（修复版）

**测试脚本：**
- `test_text_embed_onnx.py` - 文本嵌入精度测试
- `test_gpt_core_onnx.py` - GPT 核心精度测试
- `test_lm_head_onnx.py` - LM 头精度测试
- `test_conformer_onnx.py` - 条件编码器精度测试

#### 技术突破点
1. **Attention Mask 处理**：解决了 GPT Core 的 attention mask 维度不匹配问题
2. **模型结构访问**：修复了 LM Head 的模型属性访问问题
3. **多输出格式**：处理了 Conformer 的元组输出格式
4. **输入格式统一**：确保了所有模块的输入输出格式一致性

#### 参考架构
- 集成了 CosyVoice 作为子模块导出参考
- 采用与 Bert-VITS2-MNN 相似的模块化策略
- 为后续 MNN 转换奠定基础

## 项目里程碑

### 阶段一：ONNX 导出 ✅ 完成
- [x] 分析 IndexTTS 推理流程和模型结构
- [x] 设计子模块划分策略
- [x] 实现各子模块导出脚本
- [x] 建立精度验证体系
- [x] 解决 2GB 限制问题

### 阶段二：MNN 转换 🔄 待开始
- [ ] 将 ONNX 模型转换为 MNN 格式
- [ ] 实现 FP16/INT8 量化
- [ ] 性能优化和内存管理
- [ ] 端侧推理脚本开发

### 阶段三：Android 工程 🎯 目标
- [ ] Android 项目结构搭建
- [ ] JNI 接口开发
- [ ] 文本预处理端侧实现
- [ ] 音频生成和播放
- [ ] UI 界面开发

## 技术栈
- **深度学习框架**：PyTorch → ONNX → MNN
- **移动端平台**：Android (NDK + JNI)
- **语音合成**：IndexTTS (GPT + Conformer + BigVGAN2)
- **参考项目**：CosyVoice, Bert-VITS2-MNN

## 下一步计划
1. **MNN 转换**：将 4 个 ONNX 子模块转换为 MNN 格式
2. **端侧集成**：开发串联 4 个子模块的推理脚本
3. **性能优化**：实现量化、内存优化等移动端适配
4. **Android 开发**：搭建完整的移动端应用

## 文件结构
```
MobSC/
├── index-tts/                    # IndexTTS 主项目
│   ├── export_onnx/              # ONNX 导出脚本
│   │   ├── export_*.py           # 各子模块导出脚本
│   │   └── test_*.py             # 精度测试脚本
│   └── checkpoints_onnx/         # 导出的 ONNX 模型
│       ├── text_embed.onnx       # 62M
│       ├── gpt_core.onnx         # 1.8G
│       ├── lm_head.onnx          # 41M
│       └── conformer.onnx        # 156M
├── CosyVoice/                    # 参考子模块
└── PROJECT_STATUS.md             # 项目状态报告
```

---
*最后更新：2024年9月17日*
*提交：e4a8e99 - feat: 实现 IndexTTS GPT 模型子模块化 ONNX 导出*
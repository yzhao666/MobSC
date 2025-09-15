## index-tts-mnn 项目状态（Project Status）

本文件用于记录 `index-tts-mnn` 的阶段进展、问题与解决方案、关键决策与下一步计划。

关联规范：`RULES.md`（子项目）与 根目录 `RULES.md`

---

### 1) 阶段计划与完成度

- 阶段一：index-tts 纯 ONNX 推理实现（桌面端）
  - 状态：未开始 / 进行中 / 已完成（请更新）
  - 目标：导出 ONNX + ORT 推理脚本跑通端到端 WAV
  - 产出：`../checkpoints_onnx/*.onnx`、`../tools/onnx_infer.py`、一致性对齐报告

- 阶段二：MNN 转换与推理（桌面端模拟）
  - 状态：未开始 / 进行中 / 已完成（请更新）
  - 目标：MNN（FP32/FP16/INT8）转换 + 端到端推理脚本 + 评测
  - 产出：`../checkpoints_mnn/*.mnn`、`../tools/mnn_infer.*`、客观/主观评测与性能报告

- 阶段三：Android 工程实现
  - 状态：未开始 / 进行中 / 已完成（请更新）
  - 目标：端侧文本处理 + MNN 推理集成 + APK Demo
  - 产出：可安装 APK、`app/src/main/assets/` 资源、工程 README

---

### 2) 进展日志（Progress Log）

按时间倒序记录。字段建议：日期 | 模块/范围 | 类型 | 摘要 | 详情/链接 | 下一步

- 2025-__-__ | 规划 | Progress | 创建规则与阶段计划 | `RULES.md` 第14节 | 阶段一导出与ORT脚本

---

### 3) 问题与解决方案（Issues & Resolutions）

按时间倒序记录。字段建议：日期 | 模块/范围 | 问题描述 | 分析 | 解决方案 | 结论/影响

- 2025-__-__ |  |  |  |  | 

---

### 4) 决策记录（Decision Log）

按时间倒序记录。字段建议：日期 | 决策 | 选项对比 | 依据 | 影响范围 | 复盘时间点

- 2025-__-__ | 模型精度选择（FP16/INT8） | 精度/性能折中 | 阶段二评测结果 | 端侧合成音质与时延 | 阶段三前复核

---

### 5) 评测摘要（Benchmarks Summary）

阶段二产出摘要；完整细节放置于报告文件（建议 `../reports/`）

- 精度对齐：与 ONNX 基线的特征/音频误差阈值达标情况
- 客观指标：SNR、mel 相似度、WER（如适用）
- 主观指标：ABX/MOS 样本与结论
- 性能：时延（ms）、峰值内存（MB）、设备/后端（CPU/Vulkan/OpenCL）

---

### 6) 下一步（Next Actions）

- [ ] 阶段一：导出 `gpt/conformer/dvae/bigvgan` 的 ONNX，并实现 `../tools/onnx_infer.py`
- [ ] 阶段二：MNN 转换（FP32/FP16/INT8）+ `../tools/mnn_infer.*` 构建与评测
- [ ] 阶段三：集成 Android 工程与 APK Demo

---

### 7) 附录（Links & Artifacts）

- 规范：`RULES.md`（子项目）与 根目录 `RULES.md`
- 上游：`../index-tts/README.md`、`../Bert-VITS2-MNN/README.md`
- 模型目录（建议）：`../checkpoints/`、`../checkpoints_onnx/`、`../checkpoints_mnn/`
- 报告目录（建议）：`../reports/`



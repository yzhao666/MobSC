#!/usr/bin/env python3
"""
IndexTTS ONNX 推理脚本

基于导出的 ONNX 模型实现端到端推理，复现原始 PyTorch 推理流程。
"""

import argparse
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
from omegaconf import OmegaConf

from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.front import TextNormalizer, TextTokenizer


class IndexTTSOnnx:
    """基于 ONNX 模型的 IndexTTS 推理类"""
    
    def __init__(
        self, 
        config_path: str = "checkpoints/config.yaml",
        onnx_dir: str = "checkpoints_onnx",
        providers: Optional[List[str]] = None
    ):
        """
        Args:
            config_path: 配置文件路径
            onnx_dir: ONNX 模型目录
            providers: ONNXRuntime 执行提供者列表
        """
        self.cfg = OmegaConf.load(config_path)
        self.onnx_dir = onnx_dir
        
        # 设置 ONNXRuntime 提供者
        if providers is None:
            providers = ["CPUExecutionProvider"]
            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")
        
        self.providers = providers
        
        # 加载 ONNX 模型
        self._load_onnx_models()
        
        # 初始化文本处理组件
        self._init_text_processing()
        
        # 缓存相关
        self.cache_audio_prompt = None
        self.cache_cond_mel = None
        
        print(">> IndexTTS ONNX 推理器初始化完成")
    
    def _load_onnx_models(self):
        """加载所有 ONNX 模型"""
        # GPT 模型
        gpt_path = os.path.join(self.onnx_dir, "gpt.onnx")
        if os.path.exists(gpt_path):
            self.gpt_session = ort.InferenceSession(gpt_path, providers=self.providers)
            print(f">> GPT ONNX 模型加载: {gpt_path}")
        else:
            raise FileNotFoundError(f"GPT ONNX 模型未找到: {gpt_path}")
        
        # Conformer 条件编码器
        conformer_path = os.path.join(self.onnx_dir, "conformer.onnx")
        if os.path.exists(conformer_path):
            self.conformer_session = ort.InferenceSession(conformer_path, providers=self.providers)
            print(f">> Conformer ONNX 模型加载: {conformer_path}")
        else:
            print(f">> 警告: Conformer ONNX 模型未找到: {conformer_path}")
            self.conformer_session = None
        
        # DVAE 模型
        dvae_path = os.path.join(self.onnx_dir, "dvae.onnx")
        if os.path.exists(dvae_path):
            self.dvae_session = ort.InferenceSession(dvae_path, providers=self.providers)
            print(f">> DVAE ONNX 模型加载: {dvae_path}")
        else:
            print(f">> 警告: DVAE ONNX 模型未找到: {dvae_path}")
            self.dvae_session = None
        
        # BigVGAN 生成器
        bigvgan_path = os.path.join(self.onnx_dir, "bigvgan.onnx")
        if os.path.exists(bigvgan_path):
            self.bigvgan_session = ort.InferenceSession(bigvgan_path, providers=self.providers)
            print(f">> BigVGAN ONNX 模型加载: {bigvgan_path}")
        else:
            raise FileNotFoundError(f"BigVGAN ONNX 模型未找到: {bigvgan_path}")
    
    def _init_text_processing(self):
        """初始化文本处理组件"""
        # 文本标准化器
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        
        # BPE 分词器
        bpe_path = os.path.join(self.cfg.model_dir or "checkpoints", self.cfg.dataset["bpe_model"])
        self.tokenizer = TextTokenizer(bpe_path, self.normalizer)
        
        print(">> 文本处理组件初始化完成")
    
    def _extract_conditioning_mel(self, audio_prompt: str) -> np.ndarray:
        """提取参考音频的梅尔谱特征"""
        if self.cache_cond_mel is None or self.cache_audio_prompt != audio_prompt:
            # 加载音频
            audio, sr = torchaudio.load(audio_prompt)
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            
            # 重采样到 24kHz
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            
            # 提取梅尔谱特征
            cond_mel = MelSpectrogramFeatures()(audio)
            cond_mel = cond_mel.numpy()  # 转换为 numpy
            
            # 缓存
            self.cache_audio_prompt = audio_prompt
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
        
        return cond_mel
    
    def _process_text(self, text: str) -> Tuple[np.ndarray, List[List[str]]]:
        """处理文本，返回 token IDs 和分句结果"""
        # 分词
        text_tokens_list = self.tokenizer.tokenize(text)
        
        # 分句
        sentences = self.tokenizer.split_sentences(
            text_tokens_list, 
            max_tokens_per_sentence=120  # 默认值
        )
        
        # 转换为 token IDs
        all_token_ids = []
        for sent in sentences:
            token_ids = self.tokenizer.convert_tokens_to_ids(sent)
            all_token_ids.append(np.array(token_ids, dtype=np.int64))
        
        return all_token_ids, sentences
    
    def _gpt_inference_speech(self, cond_mel: np.ndarray, text_tokens: np.ndarray) -> np.ndarray:
        """GPT 模型推理生成语音 codes"""
        # 准备输入
        inputs = {
            "speech_conditioning_mel": cond_mel.astype(np.float32),
            "text_tokens": text_tokens.astype(np.int64),
            "cond_mel_lengths": np.array([cond_mel.shape[-1]], dtype=np.int64)
        }
        
        # 推理
        outputs = self.gpt_session.run(None, inputs)
        return outputs[0]  # 返回生成的 codes
    
    def _gpt_inference_latent(self, cond_mel: np.ndarray, text_tokens: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """GPT 模型推理生成潜在表示"""
        # 准备输入
        inputs = {
            "speech_conditioning_mel": cond_mel.astype(np.float32),
            "text_tokens": text_tokens.astype(np.int64),
            "codes": codes.astype(np.int64),
            "cond_mel_lengths": np.array([cond_mel.shape[-1]], dtype=np.int64)
        }
        
        # 推理
        outputs = self.gpt_session.run(None, inputs)
        return outputs[0]  # 返回潜在表示
    
    def _bigvgan_inference(self, latent: np.ndarray, cond_mel: np.ndarray) -> np.ndarray:
        """BigVGAN 模型推理生成波形"""
        # 准备输入
        inputs = {
            "latent": latent.astype(np.float32),
            "cond_mel": cond_mel.astype(np.float32)
        }
        
        # 推理
        outputs = self.bigvgan_session.run(None, inputs)
        return outputs[0]  # 返回波形
    
    def _remove_long_silence(self, codes: np.ndarray, silent_token: int = 52, max_consecutive: int = 30) -> np.ndarray:
        """移除过长的静音"""
        # 简化版本：直接截断到 stop_mel_token
        stop_mel_token = self.cfg.gpt.stop_mel_token
        
        # 找到 stop_mel_token 的位置
        stop_indices = np.where(codes == stop_mel_token)[0]
        if len(stop_indices) > 0:
            codes = codes[:stop_indices[0] + 1]
        
        return codes
    
    def infer(
        self, 
        audio_prompt: str, 
        text: str, 
        output_path: str = None,
        verbose: bool = False,
        **generation_kwargs
    ) -> str:
        """
        执行推理
        
        Args:
            audio_prompt: 参考音频路径
            text: 输入文本
            output_path: 输出音频路径
            verbose: 是否显示详细信息
            **generation_kwargs: 生成参数
        
        Returns:
            输出音频路径或音频数据
        """
        print(">> 开始 ONNX 推理...")
        start_time = time.perf_counter()
        
        if verbose:
            print(f"输入文本: {text}")
        
        # 1. 提取参考音频特征
        print(">> 提取参考音频特征...")
        cond_mel = self._extract_conditioning_mel(audio_prompt)
        if verbose:
            print(f"条件梅尔谱形状: {cond_mel.shape}")
        
        # 2. 处理文本
        print(">> 处理文本...")
        all_token_ids, sentences = self._process_text(text)
        if verbose:
            print(f"分句数量: {len(sentences)}")
            print("分句结果:", sentences)
        
        # 3. 逐句推理
        all_wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        bigvgan_time = 0
        
        for i, (token_ids, sent) in enumerate(zip(all_token_ids, sentences)):
            print(f">> 处理第 {i+1}/{len(sentences)} 句...")
            
            # 准备输入
            text_tokens = token_ids.reshape(1, -1)  # [1, seq_len]
            
            # GPT 推理生成 codes
            m_start_time = time.perf_counter()
            codes = self._gpt_inference_speech(cond_mel, text_tokens)
            gpt_gen_time += time.perf_counter() - m_start_time
            
            if verbose:
                print(f"生成的 codes 形状: {codes.shape}")
            
            # 移除长静音
            codes = self._remove_long_silence(codes[0])  # 去掉 batch 维度
            codes = codes.reshape(1, -1)  # 重新添加 batch 维度
            
            # GPT 推理生成潜在表示
            m_start_time = time.perf_counter()
            latent = self._gpt_inference_latent(cond_mel, text_tokens, codes)
            gpt_forward_time += time.perf_counter() - m_start_time
            
            if verbose:
                print(f"潜在表示形状: {latent.shape}")
            
            # BigVGAN 推理生成波形
            m_start_time = time.perf_counter()
            wav = self._bigvgan_inference(latent, cond_mel)
            bigvgan_time += time.perf_counter() - m_start_time
            
            # 后处理
            wav = np.clip(32767 * wav, -32767.0, 32767.0)
            all_wavs.append(wav)
            
            if verbose:
                print(f"波形形状: {wav.shape}, 范围: [{wav.min():.2f}, {wav.max():.2f}]")
        
        # 4. 拼接所有波形
        print(">> 拼接音频...")
        final_wav = np.concatenate(all_wavs, axis=1)
        
        # 5. 保存音频
        end_time = time.perf_counter()
        wav_length = final_wav.shape[-1] / 24000
        
        print(f">> GPT 生成时间: {gpt_gen_time:.2f} 秒")
        print(f">> GPT 前向时间: {gpt_forward_time:.2f} 秒")
        print(f">> BigVGAN 时间: {bigvgan_time:.2f} 秒")
        print(f">> 总推理时间: {end_time - start_time:.2f} 秒")
        print(f">> 生成音频长度: {wav_length:.2f} 秒")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")
        
        if output_path:
            # 转换为 torch tensor 并保存
            wav_tensor = torch.from_numpy(final_wav).type(torch.int16)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav_tensor, 24000)
            print(f">> 音频已保存到: {output_path}")
            return output_path
        else:
            # 返回音频数据
            return (24000, final_wav.astype(np.int16).T)


def main():
    parser = argparse.ArgumentParser(description="IndexTTS ONNX 推理")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml", help="配置文件路径")
    parser.add_argument("--onnx_dir", type=str, default="checkpoints_onnx", help="ONNX 模型目录")
    parser.add_argument("--voice", type=str, required=True, help="参考音频路径")
    parser.add_argument("--text", type=str, required=True, help="输入文本")
    parser.add_argument("--output", type=str, default="output_onnx.wav", help="输出音频路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    parser.add_argument("--providers", nargs="+", help="ONNXRuntime 执行提供者")
    
    args = parser.parse_args()
    
    # 创建推理器
    tts = IndexTTSOnnx(
        config_path=args.config,
        onnx_dir=args.onnx_dir,
        providers=args.providers
    )
    
    # 执行推理
    tts.infer(
        audio_prompt=args.voice,
        text=args.text,
        output_path=args.output,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
测试GPT模型导出的脚本
用于验证优化后的导出是否能够成功完成
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

def run_export_test(ckpt_path, config_path, output_dir="checkpoints_onnx", test_name="gpt_test"):
    """运行GPT导出测试"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试不同的导出配置
    test_configs = [
        {
            "name": "minimal_fp32",
            "args": ["--minimal-export", "--precision", "fp32", "--max-mel-frames", "64", "--max-text-len", "16"],
            "description": "最小化FP32导出"
        },
        {
            "name": "small_fp32", 
            "args": ["--precision", "fp32", "--max-mel-frames", "128", "--max-text-len", "32"],
            "description": "小尺寸FP32导出"
        },
        {
            "name": "small_fp16",
            "args": ["--precision", "fp16", "--max-mel-frames", "128", "--max-text-len", "32"], 
            "description": "小尺寸FP16导出"
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"测试配置: {config['description']}")
        print(f"{'='*60}")
        
        output_path = os.path.join(output_dir, f"{test_name}_{config['name']}.onnx")
        
        # 构建命令
        cmd = [
            sys.executable, "export_onnx/export_gpt.py",
            "--ckpt", ckpt_path,
            "--config", config_path,
            "--out", output_path
        ] + config["args"]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # 运行导出命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30分钟超时
                cwd=os.getcwd()
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ 导出成功! 耗时: {duration:.2f}秒")
                print(f"输出文件: {output_path}")
                
                # 检查文件大小
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    print(f"文件大小: {file_size:.2f}MB")
                
                results.append({
                    "config": config["name"],
                    "status": "success",
                    "duration": duration,
                    "file_size": file_size if os.path.exists(output_path) else 0
                })
            else:
                print(f"❌ 导出失败! 返回码: {result.returncode}")
                print(f"错误输出: {result.stderr}")
                results.append({
                    "config": config["name"],
                    "status": "failed",
                    "duration": duration,
                    "error": result.stderr
                })
                
        except subprocess.TimeoutExpired:
            print(f"⏰ 导出超时! (超过30分钟)")
            results.append({
                "config": config["name"],
                "status": "timeout",
                "duration": 1800
            })
        except Exception as e:
            print(f"💥 导出异常: {e}")
            results.append({
                "config": config["name"],
                "status": "error",
                "error": str(e)
            })
    
    # 打印测试结果汇总
    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    
    for result in results:
        status_icon = {
            "success": "✅",
            "failed": "❌", 
            "timeout": "⏰",
            "error": "💥"
        }.get(result["status"], "❓")
        
        print(f"{status_icon} {result['config']}: {result['status']}")
        if result["status"] == "success":
            print(f"   耗时: {result['duration']:.2f}秒, 文件大小: {result['file_size']:.2f}MB")
        elif "error" in result:
            print(f"   错误: {result['error'][:100]}...")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="测试GPT模型导出")
    parser.add_argument("--ckpt", type=str, default="checkpoints/gpt.pth", help="GPT模型检查点路径")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml", help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default="checkpoints_onnx", help="输出目录")
    parser.add_argument("--test-name", type=str, default="gpt_test", help="测试名称")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.ckpt):
        print(f"❌ 检查点文件不存在: {args.ckpt}")
        return 1
        
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return 1
    
    print("🚀 开始GPT模型导出测试...")
    print(f"检查点: {args.ckpt}")
    print(f"配置: {args.config}")
    print(f"输出目录: {args.output_dir}")
    
    results = run_export_test(args.ckpt, args.config, args.output_dir, args.test_name)
    
    # 统计成功/失败
    success_count = sum(1 for r in results if r["status"] == "success")
    total_count = len(results)
    
    print(f"\n📊 测试完成: {success_count}/{total_count} 成功")
    
    if success_count > 0:
        print("🎉 至少有一个配置导出成功!")
        return 0
    else:
        print("😞 所有配置都失败了")
        return 1

if __name__ == "__main__":
    sys.exit(main())

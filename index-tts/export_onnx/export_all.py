import argparse
import os
import subprocess


def run(cmd):
    print("$ ", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gpt_ckpt = os.path.join(args.model_dir, "gpt.pth")
    conformer_ckpt = os.path.join(args.model_dir, "conformer.pth")
    dvae_ckpt = os.path.join(args.model_dir, "dvae.pth")
    bigvgan_ckpt = os.path.join(args.model_dir, "bigvgan_generator.pth")

    # 优先模块方式（在 index-tts 目录下）
    try:
        run(["python", "-m", "export_onnx.export_gpt", "--ckpt", gpt_ckpt, "--config", args.config, "--out", os.path.join(args.out_dir, "gpt.onnx"), "--opset", str(args.opset)])
        run(["python", "-m", "export_onnx.export_conformer", "--ckpt", conformer_ckpt, "--config", args.config, "--out", os.path.join(args.out_dir, "conformer.onnx"), "--opset", str(args.opset)])
        run(["python", "-m", "export_onnx.export_dvae", "--ckpt", dvae_ckpt, "--config", args.config, "--out", os.path.join(args.out_dir, "dvae.onnx"), "--opset", str(args.opset)])
        run(["python", "-m", "export_onnx.export_bigvgan", "--ckpt", bigvgan_ckpt, "--config", args.config, "--out", os.path.join(args.out_dir, "bigvgan.onnx"), "--opset", str(args.opset)])
    except Exception:
        # 回退到根目录直跑
        run(["python", os.path.join("index-tts", "export_onnx", "export_gpt.py"), "--ckpt", gpt_ckpt, "--config", args.config, "--out", os.path.join(args.out_dir, "gpt.onnx"), "--opset", str(args.opset)])
        run(["python", os.path.join("index-tts", "export_onnx", "export_conformer.py"), "--ckpt", conformer_ckpt, "--config", args.config, "--out", os.path.join(args.out_dir, "conformer.onnx"), "--opset", str(args.opset)])
        run(["python", os.path.join("index-tts", "export_onnx", "export_dvae.py"), "--ckpt", dvae_ckpt, "--config", args.config, "--out", os.path.join(args.out_dir, "dvae.onnx"), "--opset", str(args.opset)])
        run(["python", os.path.join("index-tts", "export_onnx", "export_bigvgan.py"), "--ckpt", bigvgan_ckpt, "--config", args.config, "--out", os.path.join(args.out_dir, "bigvgan.onnx"), "--opset", str(args.opset)])

    print("[OK] All ONNX models exported to:", args.out_dir)


if __name__ == "__main__":
    main()



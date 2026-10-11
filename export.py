import os
import argparse
import torch


def parse_shape(s):
    """
    解析 "1,3,224,224" / "1 3 224 224" / "(1,3,224,224)" 为 (1,3,224,224)
    """
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    s = s.strip().strip("()")
    for sep in (",", " "):
        if sep in s:
            return tuple(int(p) for p in s.split(sep) if p != "")
    return (int(s),)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export PyTorch model to ONNX format')
    parser.add_argument("--weight", default='./run/best.pth', type=str,
                        help='Path to the PyTorch model file (full model object)')
    parser.add_argument("--input_size", default="1,1,224,224", type=str,
                        help='Input size, e.g. "1,1,224,224"')
    parser.add_argument("--output", default=None, type=str,
                        help='Output ONNX path; default: same dir/name as weight')
    parser.add_argument("--device", default='cpu', type=str,
                        help='Device to use (cuda or cpu)')
    parser.add_argument("--onnx_version", default=12, type=int,
                        help='ONNX opset version (default 12)')
    parser.add_argument("--dynamic", action='store_true',
                        help='Enable dynamic axes for input and output')
    parser.add_argument("--verbose", action='store_true',
                        help='Enable verbose output during export')
    parser.add_argument("--constant_folding", action='store_true',
                        help='Enable constant folding optimization')
    args = parser.parse_args()

    # ---------- 设备 ----------
    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    elif args.device.isdigit():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device(args.device)

    # ---------- 解析输入尺寸 ----------
    input_size = parse_shape(args.input_size)
    if len(input_size) != 4:
        raise ValueError(
            f"--input_size 必须是 4 个维度 (batch, channel, H, W)，"
            f"当前为 {input_size}")

    # ---------- 加载完整模型对象 ----------
    obj = torch.load(args.weight, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj:
        obj = obj['state_dict']

    if not isinstance(obj, torch.nn.Module):
        raise ValueError(
            f"{args.weight} 不是完整模型对象。"
            f"请确保 train.py 用 torch.save(model) 保存整模型。")

    model = obj.to(device).eval()

    # ---------- 推断任务数（多任务模型）/ 输出名 ----------
    # MultiTaskModel 有 .heads 属性；单输出模型没有
    if hasattr(model, 'heads'):
        num_outputs = len(model.heads)
        output_names = [f"output{i}" for i in range(num_outputs)]
    else:
        num_outputs = 1
        output_names = ["output0"]

    print(f"模型输出数: {num_outputs}，输出名: {output_names}")

    # ---------- 构造 dummy input ----------
    input_tensor = torch.randn(*input_size, device=device)

    # ---------- 输出路径（这就是之前缺失的那段） ----------
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.weight)[0]
        output_path = base + ".onnx"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)),
                exist_ok=True)

    # ---------- 动态轴 ----------
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input0": {0: "batch_size", 2: "height", 3: "width"},
        }
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

    # ---------- 导出 ----------
    with torch.inference_mode():
        torch.onnx.export(
            model,
            input_tensor,
            output_path,
            input_names=["input0"],
            output_names=output_names,
            opset_version=args.onnx_version,
            dynamic_axes=dynamic_axes,
            do_constant_folding=args.constant_folding,
            verbose=args.verbose,
            keep_initializers_as_inputs=False,
        )

    # ---------- 校验 ----------
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型校验通过。")
    except ImportError:
        print("未安装 onnx，跳过模型校验。")
    except Exception as e:
        print(f"[警告] ONNX 校验失败：{e}")

    print(f"Successfully exported ONNX model to: {output_path}")
    print(f"ONNX opset version used: {args.onnx_version}")
    print(f"Input size: {input_size}")
    if args.dynamic:
        print("Dynamic axes enabled for batch size and spatial dimensions")
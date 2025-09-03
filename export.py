import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX format with configurable parameters')
    parser.add_argument("--weight", default='.\\run\\best.pth', type=str,
                      help='Path to the PyTorch weight file')
    parser.add_argument("--input_size", default=(1, 1, 224, 224), type=tuple,
                      help='Input size tuple (batch, channel, height, width)')
    parser.add_argument("--device", default='cuda', type=str,
                      help='Device to use for inference (cuda or cpu)')
    parser.add_argument("--onnx_version", default=12, type=int,
                      help='ONNX opset version to use (default: 12)')
    parser.add_argument("--dynamic", action='store_true',
                      help='Enable dynamic axes for input and output')
    parser.add_argument("--verbose", action='store_true',
                      help='Enable verbose output during export')
    parser.add_argument("--constant_folding", action='store_true',
                      help='Enable constant folding optimization')
    
    args = parser.parse_args()

    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = torch.load(args.weight, map_location='cpu')
    model.to(device)
    model.eval()

    # Create dummy input
    input_tensor = torch.rand(args.input_size).to(device)
    
    # Prepare output path
    output_path = args.weight.split('.')[0] + ".onnx"
    
    # Set up dynamic axes if enabled
    dynamic_axes = None
    if args.dynamic:
        # Allow batch size and spatial dimensions to be dynamic
        dynamic_axes = {
            "input0": {0: "batch_size", 2: "height", 3: "width"},
            "output0": {0: "batch_size"}
        }
    
    # Export model to ONNX
    torch.onnx.export(
        model,                          # Model to export
        input_tensor,                   # Model input tensor
        output_path,                    # Output file path
        input_names=["input0"],         # Input tensor names
        output_names=["output0"],       # Output tensor names
        opset_version=args.onnx_version, # ONNX opset version
        dynamic_axes=dynamic_axes,      # Dynamic axes configuration
        do_constant_folding=args.constant_folding, # Constant folding optimization
        verbose=args.verbose,           # Verbose output
        keep_initializers_as_inputs=False # Whether to keep initializers as inputs
    )

    print(f"Successfully exported ONNX model to: {output_path}")
    print(f"ONNX opset version used: {args.onnx_version}")
    if args.dynamic:
        print("Dynamic axes enabled for batch size and spatial dimensions")

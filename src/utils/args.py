from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser("PyTorch vs ONNX vs TensorRT inference comparison")
    parser.add_argument(
        "--device",
        type=str,
        help="Inference device ('cpu' or 'cuda'). cuda available for all engines, cpu not available for TensorRT.",
        default="cuda",
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="Inference engine ('trt', 'onnx' or 'pytorch').",
        default="trt",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to use for measurements. It defines the pytorch module and engine config loader functions (look in src/load.py).",
        default="resnet152",
    )
    return parser.parse_args()

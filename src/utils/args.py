import ast
from argparse import ArgumentParser, Namespace

from typing import List, Optional, Union, Tuple

def parse_example_shapes(
    example_shapes_str: Optional[str],
) -> Optional[List[Tuple[int, ...]]]:
    if example_shapes_str is None:
        return None
    example_shapes = ast.literal_eval(example_shapes_str)
    if isinstance(example_shapes, tuple):
        example_shapes = list(example_shapes)
    for i, example_shape in enumerate(example_shapes):
        if isinstance(example_shape, int):
            example_shapes[i] = (example_shape,)
    return example_shapes


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
    parser.add_argument(
        "--num_iter",
        type=int,
        help="Number of iterations for measurements.",
        default=1000,
    )
    parser.add_argument(
        "--num_warmup_iter",
        type=int,
        help="Number of warmup iterations (not included in measurements statistics)",
        default=100,
    )
    parser.add_argument(
        "--example_shapes",
        type=str,
        help='Example inputs shapes in form "([dim_00, ..., dim_0N], ..., [dim_N0, ..., dim_NN])"',
        default=None,
    )
    args = parser.parse_args()
    args.example_shapes = parse_example_shapes(args.example_shapes)
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    print(args.example_shapes, type(args.example_shapes))

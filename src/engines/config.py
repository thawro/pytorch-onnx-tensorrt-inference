from dataclasses import asdict, dataclass

import numpy as np
import yaml
from dacite import from_dict


@dataclass
class OptimizationShapes:
    min: list[int]
    opt: list[int]
    max: list[int]

    def to_dict(self):
        return asdict(self)


@dataclass
class InputShape:
    dims_names: list[str]
    runtime: list[int]
    example: list[int]
    optimization: OptimizationShapes


@dataclass
class OutputShape:
    runtime: list[int]
    dims_names: list[str]


@dataclass
class Input:
    name: str
    dtype_str: str
    shapes: InputShape

    def __post_init__(self):
        self.dtype = getattr(np, self.dtype_str)

    @property
    def is_dynamic(self) -> bool:
        return -1 in self.shapes.runtime


@dataclass
class Output:
    name: str
    shape: OutputShape

    @property
    def is_dynamic(self) -> bool:
        return -1 in self.shape.runtime


@dataclass
class EngineConfig:
    name: str
    inputs: list[Input]
    outputs: list[Output]

    def __post_init__(self):
        self.onnx_filename = f"{self.name}.onnx"
        self.trt_filename = f"{self.name}.engine"

    def save_to_yaml(self, filepath: str):
        with open(filepath, "w") as outfile:
            yaml.dump(asdict(self), outfile)

    @property
    def has_dynamic_input(self) -> bool:
        return any([inp.is_dynamic for inp in self.inputs])

    @classmethod
    def from_yaml(cls, filepath: str) -> "EngineConfig":
        with open(filepath) as file:
            data = yaml.safe_load(file)
        return from_dict(data_class=cls, data=data)

    @property
    def example_inputs_shapes_str(self) -> str:
        return str([inp.shapes.example for inp in self.inputs])

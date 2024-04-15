# PyTorch - ONNX - TensorRT inference comparison
Inference time comparison between **`PyTorch`**, **`ONNX`** and **`TensorRT`** engines

# How to run measurements

> **NOTE:** By default for latency plots outliers are replaced with NaNs

1. Implement your custom model module for config and PyTorch model loading

The custom modules must be placed in `models/<custom_model_name>` and there must be `config.yaml` and `load.py` files present:
* `config.yaml` - config for engines (ONNX and TensorRT) building ([how to form a config](#model-config))
* `load.py` - a file with  an *Engine Loader* class definition - *Engine Loader* must implement:
  * `load_example_inputs` - method which returns example model inputs (before preprocessing) as `list[numpy.array]`
  * `load_pytorch_module` - method which returns your model as `torch.nn.Module`
  * `name` attribute (used to get proper loader when calling bash scripts (`model_name` key))
  
2. Wrap your *Engine Loader* class with `@register_model` decorator (imported from from `src/utils/utils.py`)

3. Add import statement in `src/load.py` (so decorator registers the model)

4. Set up measurements configuration in `run_all_measurements.sh` script:

   * `num_iter` = number of measurements iterations
   * `num_warmup_iter` = number of warmup iterations
   * `model_name` = *Engine Loader* name
   * `experiments_example_shapes` = string representation of example input shapes to perform experiments on, eg. "[(224,224,3)]" "[(336,336,3)]" "[(448,448,3)]" will perform 3 separate measurements for input shapes (224,224,3), (336,336,3) and (448,448,3)

5. Run all measurements:

```bash
bash run_all_measurements.sh
```

6. Analyse results stored in `measurements_results/<model_name>` directory


# Model config
Model config is used to parse model from `PyTorch` to `ONNX` and from `ONNX` to `TensorRRT`
```yaml
name: Name of the model used to define model names during files handling
inputs: # list of input definitions
  - name: Name of the input 
    dtype_str: Data type (string representation) of the input, eg. float32
    shapes: # Input shape information
      dims_names: Name for each dim, eg. [batch, C, H, W]
      example: Dimensions of example input (before preprocessing) used for tests, eg. [224, 224, 3]
      runtime: Runtime dimensions (use -1 for dynamic dim), eg. [1, 3, -1, -1]
      optimization: # Dimensions used in TensorRT profiling
        min: Minimum dimensions, eg. [1, 3, 128, 128]
        opt: Optimum dimensions, eg. [1, 3, 256, 256]
        max: Maximum dimensions, eg. [1, 3, 512, 512]
outputs: # list of output definitions
  - name: Name of the output
    shape: # Output shape information
      runtime: Runtime dimensions, eg. [1, 1000]
      dims_names: Name for each dim, eg. [batch, probs]
```

# Example - `ResNet50` architecture
Model config:
```yaml
name: resnet50
inputs:
  - name: image
    dtype_str: float32
    shapes:
      dims_names: [batch_size, channels, height, width]
      example: [224, 224, 3]
      runtime: [1, 3, -1, -1]
      optimization:
        min: [1, 3, 128, 128]
        opt: [1, 3, 256, 256]
        max: [1, 3, 512, 512]
outputs:
  - name: probs
    shape:
      runtime: [1, 1000]
      dims_names: [batch_size, probs]
```

## CUDA

### Latency
![cuda_latency](./measurements_results/resnet50/plots/[(224,224,3)]/cuda_time_measurements.jpg)

### Memory (VRAM) [mb] 
![gpu_vram](./measurements_results/resnet50/plots/[(224,224,3)]/gpu_0_mb_measurements.jpg)

### Utilisation [%] 
![gpu_util](./measurements_results/resnet50/plots/[(224,224,3)]/gpu_0_pct_util_measurements.jpg)

## CPU

### Latency
![cpu_latency](./measurements_results/resnet50/plots/[(224,224,3)]/cpu_time_measurements.jpg)

### Memory (RAM) [mb] 
![cpu_ram](./measurements_results/resnet50/plots/[(224,224,3)]/cpu_mb_measurements.jpg)

### Utilisation [%] 
![cpu_util](./measurements_results/resnet50/plots/[(224,224,3)]/cpu_pct_util_measurements.jpg)
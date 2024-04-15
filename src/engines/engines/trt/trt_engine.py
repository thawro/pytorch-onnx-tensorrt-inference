import logging

import numpy as np
import tensorrt as trt
from cuda import cudart

from src.engines.engines.base import BaseInferenceEngine
from src.engines.engines.trt.memory import HostDeviceMem, allocate_buffers, free_buffers
from src.engines.engines.trt.utils import TRT_MAJOR_VERSION, cuda_call
from src.monitoring.time import measure_time

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


dtypes = {np.float32: trt.float32}


def _do_inference_base(inputs, outputs, stream, execute_async_func):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    for inp in inputs:
        cuda_call(
            cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)
        )

    # Run inference.
    execute_async_func()
    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost

    for out in outputs:
        cuda_call(
            cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)
        )

    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))
    # Return only the host outputs.
    return [out.host for out in outputs]


def do_inference(
    context: trt.IExecutionContext,
    engine: trt.ICudaEngine,
    bindings: list[int],
    inputs: list[HostDeviceMem],
    outputs: list[HostDeviceMem],
    stream,
):
    # This function is generalized for multiple inputs/outputs.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def execute_async_func():
        if TRT_MAJOR_VERSION >= 10:
            context.execute_async_v3(stream_handle=stream)
        else:
            context.execute_async_v2(bindings=bindings, stream_handle=stream)

    # Setup context tensor address.
    for i in range(engine.num_io_tensors):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    return _do_inference_base(inputs, outputs, stream, execute_async_func)


class TensorRTInferenceEngine(BaseInferenceEngine):
    engine: trt.ICudaEngine
    context: trt.IExecutionContext
    name: str = "TensorRT"

    def allocate_buffers(self):
        # Allocate buffers and create a CUDA stream.
        self.inputs, self.outputs, self.bindings = allocate_buffers(
            self.engine, self.context, profile_idx=0
        )

    def create_context(self) -> trt.IExecutionContext:
        # Contexts are used to perform inference.
        logging.info("-> Creating trt.IExecutionContext")
        context = self.engine.create_execution_context()
        self.context = context
        return context

    def create_stream(self):
        logging.info("-> Creating Stream handle (pointer)")
        stream = cuda_call(cudart.cudaStreamCreate())
        self.stream = stream
        return stream

    def load_engine_from_onnx(self, dirpath: str):
        logging.info("-> Loading trt.ICudaEngine from ONNX file")
        builder = trt.Builder(TRT_LOGGER)
        if TRT_MAJOR_VERSION >= 10:
            network_flags = 0
        else:
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        network = builder.create_network(network_flags)

        config = builder.create_builder_config()
        if self.cfg.has_dynamic_input:
            logging.info("Model config has dynamic runtime shape.")
            logging.info("-> Creating Optimization profile")
            profile = builder.create_optimization_profile()
            for input in self.cfg.inputs:
                profile.set_shape(input.name, **input.shapes.optimization.to_dict())
            config.add_optimization_profile(profile)

        parser = trt.OnnxParser(network, TRT_LOGGER)

        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(1))
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(f"{dirpath}/{self.cfg.onnx_filename}", "rb") as model_file:
            logging.info("-> Parsing ONNX file")
            success = parser.parse(model_file.read())
            if not success:
                logging.error("Failed to parse the ONNX file.")
                msgs = ""
                for error in range(parser.num_errors):
                    msgs += f"{parser.get_error(error)}\n"
                e = Exception(msgs)
                logging.exception(e)
                raise e
        engine_bytes = builder.build_serialized_network(network, config)
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        logging.info("-> ONNX parsed successfully")

    def load_engine_from_trt(self, dirpath: str):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(f"{dirpath}/{self.cfg.trt_filename}", "rb") as model_file:
            self.engine = runtime.deserialize_cuda_engine(model_file.read())

    def save_engine_to_trt(self, dirpath: str):
        with open(f"{dirpath}/{self.cfg.trt_filename}", "wb") as engine_file:
            engine_file.write(self.engine.serialize())

    def move_inputs_to_device(self, inputs: list[np.ndarray]):
        for i, inp in enumerate(inputs):
            np.copyto(self.inputs[i].host, inp.ravel())

    @measure_time(time_unit="ms", name="TensorRT")
    def inference(
        self, inputs: list[np.ndarray], context: trt.IExecutionContext | None = None
    ) -> list[np.ndarray]:
        if context is None:
            context = self.context

        preprocessed_inputs = self.preprocess_inputs(inputs)
        self.move_inputs_to_device(preprocessed_inputs)

        outputs = do_inference(
            context=context,
            engine=self.engine,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
        )
        return outputs

    def free_buffers(self):
        free_buffers(self.inputs, self.outputs, self.stream)

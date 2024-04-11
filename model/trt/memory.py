import ctypes
import logging

import numpy as np
import tensorrt as trt
from cuda import cudart

from .utils import cuda_call


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""

    def __init__(self, size: int, dtype: np.dtype | None = None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: np.ndarray | bytes):
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                )
            np.copyto(self.host[: data.size], data.flat, casting="safe")
        else:
            assert self.host.dtype == np.uint8
            self.host[: self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


def allocate_buffers(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext | None = None,
    profile_idx: int | None = None,
):
    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    # If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
    logging.info("-> Allocating Buffers")
    inputs = []
    outputs = []
    bindings = []

    num_tensors = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(num_tensors)]
    for name in tensor_names:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        if context is not None:
            shape = context.get_tensor_shape(name)
        elif profile_idx is not None:
            shape = engine.get_tensor_profile_shape(name, profile_idx)[1]  # opt shape
        else:
            shape = engine.get_tensor_shape(name)
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(
                f"Binding {name} has dynamic shape, " + "but no profile was specified."
            )
        size = trt.volume(shape)
        trt_type = engine.get_tensor_dtype(name)

        # Allocate host and device buffers
        if trt.nptype(trt_type):
            dtype = np.dtype(trt.nptype(trt_type))
            bindingMemory = HostDeviceMem(size, dtype)
        else:  # no numpy support: create a byte array instead (BF16, FP8, INT4)
            size = int(size * trt_type.itemsize)
            bindingMemory = HostDeviceMem(size)

        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device))

        # Append to the appropriate list.
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            mode = "Input"
            inputs.append(bindingMemory)
        else:
            mode = "Output"
            outputs.append(bindingMemory)
        logging.info(
            f"-> Allocated Buffer {name} (mode={mode}, shape={shape}, dtype={dtype}, size={size})"
        )
    return inputs, outputs, bindings


def free_buffers(
    inputs: list[HostDeviceMem],
    outputs: list[HostDeviceMem],
    stream: cudart.cudaStream_t,
):
    # Frees the resources allocated in allocate_buffers
    logging.info("-> Freeing buffers")
    for mem in inputs:
        mem.free()
    logging.info("-> Freed inputs buffers")
    for mem in outputs:
        mem.free()
    logging.info("-> Freed outputs buffers")
    cuda_call(cudart.cudaStreamDestroy(stream))
    logging.info("-> Freed stream buffer")


def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    # Wrapper for cudaMemcpy which infers copy size and does error checking
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
    )


def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    # Wrapper for cudaMemcpy which infers copy size and does error checking
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
    )

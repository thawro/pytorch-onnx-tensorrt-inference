import ctypes
import logging

import numpy as np
import tensorrt as trt

import pycuda.autoinit
import pycuda.driver as cuda

from typing import List, Optional, Union


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""

    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        dtype = dtype or np.dtype(np.uint8)
        host_mem = cuda.pagelocked_empty(size, dtype)
        nbytes = host_mem.nbytes

        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda.mem_alloc(nbytes)
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: Optional[np.ndarray]):
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
        pass
        # TODO: free device memory with pycuda
        # TODO: free host memory with pycuda




def allocate_buffers(
    engine: trt.ICudaEngine,
    context: Optional[trt.IExecutionContext] = None,
    profile_idx: Optional[int] = None
):
    logging.info("-> Allocating Buffers")
    inputs = []
    outputs = []
    bindings = []

    for name in engine:
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
    inputs: List[HostDeviceMem],
    outputs: List[HostDeviceMem],
    stream: cuda.Stream,
):
    # Frees the resources allocated in allocate_buffers
    logging.info("-> Freeing buffers")
    for mem in inputs:
        mem.free()
    logging.info("-> Freed inputs buffers")
    for mem in outputs:
        mem.free()
    logging.info("-> Freed outputs buffers")
    # TODO: destroy stream
    logging.info("-> Freed stream buffer")

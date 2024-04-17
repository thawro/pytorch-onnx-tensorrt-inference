import ctypes
import logging

import numpy as np
import tensorrt as trt

import pycuda.autoinit
import pycuda.driver as cuda

from typing import List, Optional, Union


class HostDeviceMem:
	"""Pair of host and device memory, where the host memory is wrapped in a numpy array"""

	def __init__(self, size: int, trt_dtype):
		host_mem = cuda.pagelocked_empty(size, trt_dtype)
		device_mem = cuda.mem_alloc(host_mem.nbytes)
		self.host = host_mem
		self.device = device_mem
		self.nbytes = host_mem.nbytes

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
		binding_idx = engine.get_binding_index(name)
		if context is not None:
		    # shape = context.get_shape(binding_idx)
		    shape = context.get_binding_shape(binding_idx)
		elif profile_idx is not None:
		    shape = engine.get_profile_shape(profile_idx, binding_idx)[1]  # opt shape
		else:
		    shape = engine.get_binding_shape(binding_idx)
		shape_valid = np.all([s >= 0 for s in shape])
		if not shape_valid and profile_idx is None:
		    raise ValueError(
		        f"Binding {name} has dynamic shape, " + "but no profile was specified."
		    )
		size = trt.volume(shape)
		trt_dtype = trt.nptype(engine.get_binding_dtype(binding_idx))

		# Allocate host and device buffers
		bindingMemory = HostDeviceMem(size, trt_dtype)
		
		# Append the device buffer to device bindings.
		bindings.append(int(bindingMemory.device))

		# Append to the appropriate list.
		if engine.binding_is_input(name):
		    mode = "Input"
		    inputs.append(bindingMemory)
		else:
		    mode = "Output"
		    outputs.append(bindingMemory)
		logging.info(
		    f"-> Allocated Buffer {name} (mode={mode}, shape={shape}, dtype={trt_dtype}, size={size})"
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

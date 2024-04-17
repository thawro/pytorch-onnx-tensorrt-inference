import tensorrt as trt

TRT_MAJOR_VERSION = int(trt.__version__.split(".")[0])


def GiB(val):
    return val * 1 << 30

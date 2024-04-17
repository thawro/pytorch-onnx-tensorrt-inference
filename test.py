from jtop import jtop

with jtop() as jetson:
	if jetson.ok():
		stats = jetson.stats
		gpu_memory = stats['RAM']
		temp_CPU = stats['Temp CPU']
		temp_GPU = stats['Temp GPU']
		power_cur = stats['power cur']
		gpu_util = jetson.gpu['val']


import psutil

cpu_util = psutil.cpu_percent()
cpu_memory = psutil.virtual_memory()

print(cpu_util, cpu_memory.used / 1e6)
print(gpu_util, gpu_memory/1e3)



python3 run_measurements.py --device="cuda" --engine="ONNX" --model_name="resnet18" --num_iter=100 --num_warmup_iter=10 --example_shapes="[(224,224,3)]"


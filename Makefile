run_measurements:
	# python3 prepare_engine_files.py
	python3 run_measurements.py --device="cuda" --engine="TensorRT"
	python3 run_measurements.py --device="cuda" --engine="ONNX"
	python3 run_measurements.py --device="cuda" --engine="PyTorch"
	python3 run_measurements.py --device="cpu" --engine="ONNX"
	python3 run_measurements.py --device="cpu" --engine="PyTorch"
	python3 plot_measurements.py


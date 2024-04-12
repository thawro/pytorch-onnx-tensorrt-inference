measurements:
	# python3 prepare_engine_files.py
	sleep 1
	python3 run_measurements.py --device="cuda" --engine="TensorRT"
	sleep 1
	python3 run_measurements.py --device="cuda" --engine="ONNX"
	sleep 1
	python3 run_measurements.py --device="cuda" --engine="PyTorch"
	sleep 1
	python3 run_measurements.py --device="cpu" --engine="ONNX"
	sleep 1
	python3 run_measurements.py --device="cpu" --engine="PyTorch"
	sleep 1
	python3 plot_measurements.py


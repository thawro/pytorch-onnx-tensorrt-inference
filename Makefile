engines:
	python3 prepare_engine_files.py --model_name=$(model_name)

measurements:
	python3 run_measurements.py --device="cuda" --engine="PyTorch" --model_name=$(model_name) --num_iter=$(num_iter) --num_warmup_iter=$(num_warmup_iter) --example_shapes="$(example_shapes)"
	python3 run_measurements.py --device="cuda" --engine="ONNX" --model_name=$(model_name) --num_iter=$(num_iter) --num_warmup_iter=$(num_warmup_iter) --example_shapes="$(example_shapes)"
	python3 run_measurements.py --device="cuda" --engine="TensorRT" --model_name=$(model_name) --num_iter=$(num_iter) --num_warmup_iter=$(num_warmup_iter) --example_shapes="$(example_shapes)"
	python3 run_measurements.py --device="cpu" --engine="PyTorch" --model_name=$(model_name) --num_iter=$(num_iter) --num_warmup_iter=$(num_warmup_iter) --example_shapes="$(example_shapes)"
	python3 run_measurements.py --device="cpu" --engine="ONNX" --model_name=$(model_name) --num_iter=$(num_iter) --num_warmup_iter=$(num_warmup_iter) --example_shapes="$(example_shapes)"
	
plots:
	python3 plot_measurements.py --model_name=$(model_name)
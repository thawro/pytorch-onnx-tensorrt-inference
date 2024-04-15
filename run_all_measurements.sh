#!/bin/bash
set -e # make sure to stop when any of the commands raises Exception

num_iter=1000
num_warmup_iter=100

# Define model name and example shapes for which to run measurements
model_name="resnet50"
experiments_example_shapes=("[(224,224,3)]" "[(336,336,3)]" "[(448,448,3)]")
# experiments_example_shapes=("[(224,224,3)]")


make engines model_name="$model_name"

for example_shapes in "${experiments_example_shapes[@]}"; do
    make measurements model_name="$model_name" num_iter=$num_iter num_warmup_iter=$num_warmup_iter example_shapes="$example_shapes"
done

make plots model_name="$model_name"
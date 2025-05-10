#!/usr/bin/env bash
# filepath: /home/gpudual/bhklab/jermiah/projects/simple-ml-project/workflow/scripts/train.sh
set -e

# Create a timestamp-based model name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_NAME="mnist_classifier_${TIMESTAMP}"

# Train an MNIST classifier using the CLI
pixi run python -m simple_ml_project.cli.commands train-model \
    --data-dir data/rawdata \
    --out-dir data/results \
    --epochs 10 \
    --lr 0.001 \
    --batch-size 256 \
    --model-name "$MODEL_NAME" \
    --save-checkpoints

# Log the results
echo "Training completed successfully."
echo "Model saved as: $MODEL_NAME"
echo "Results available in: data/results"
echo "To run inference with this model:"
echo "pixi run python -m simple_ml_project.cli.commands predict --model-path data/results/models/${MODEL_NAME}_best.pt --image-path <your-image-path>"
#!/usr/bin/env bash
# filepath: /home/gpudual/bhklab/jermiah/projects/simple-ml-project/workflow/scripts/infer.sh
set -e

# Run inference on a single PNG image using the trained model
INPUT_IMAGE="$1"
MODEL_PATH="${2:-data/results/models/mnist_classifier_best.pt}"
OUTPUT_PATH="${3:-inference_result_$(date +"%Y%m%d_%H%M%S").png}"

if [[ ! -f "$INPUT_IMAGE" ]]; then
	echo "❌ Input image not found: $INPUT_IMAGE"
	exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
	echo "❌ Model not found: $MODEL_PATH"
	exit 1
fi

echo "Running inference on $INPUT_IMAGE using model $MODEL_PATH"

# Run inference using the CLI
pixi run python -m simple_ml_project.cli.commands predict \
	--model-path "$MODEL_PATH" \
    --image-path "$INPUT_IMAGE" \
	--output-path "$OUTPUT_PATH"

echo "Inference completed successfully."
echo "Result saved to: $OUTPUT_PATH"
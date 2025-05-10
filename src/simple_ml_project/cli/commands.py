# src/simple_ml_project/cli/commands.py

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import click

from simple_ml_project.inference import run_inference
from simple_ml_project.train import train


# Configure logging
def setup_logging(verbose=False):
	"""Configure logging based on verbosity level."""
	level = logging.DEBUG if verbose else logging.INFO

	# Configure root logger
	logging.basicConfig(
		level=level,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		handlers=[
			logging.StreamHandler(sys.stdout),
		],
	)

	# Return logger for CLI
	return logging.getLogger('simple_ml_project.cli')


# Pass context between commands
@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
	"""Simple ML Project - MNIST Classifier CLI.

	A robust machine learning project for MNIST digit classification.
	"""
	# Ensure context object exists
	ctx.ensure_object(dict)

	# Set up logging
	ctx.obj['logger'] = setup_logging(verbose)

	# Store verbosity setting
	ctx.obj['verbose'] = verbose

	if verbose:
		ctx.obj['logger'].debug('Verbose mode enabled')


@cli.command()
@click.option(
	'--data-dir',
	type=click.Path(exists=True, file_okay=False, path_type=Path),
	default=Path('data/rawdata'),
	help='Directory containing the input data',
)
@click.option(
	'--out-dir',
	type=click.Path(file_okay=False, path_type=Path),
	default=Path('data/results'),
	help='Directory to save model and results',
)
@click.option(
	'--epochs',
	type=click.IntRange(min=1),
	default=5,
	help='Number of training epochs',
)
@click.option(
	'--lr',
	type=float,
	default=1e-3,
	help='Learning rate',
)
@click.option(
	'--batch-size',
	type=click.IntRange(min=1),
	default=256,
	help='Batch size for training',
)
@click.option(
	'--model-name',
	type=str,
	default=None,
	help='Name for the saved model (default: mnist_classifier_{timestamp})',
)
@click.option(
	'--save-checkpoints',
	is_flag=True,
	help='Save intermediate checkpoints during training',
)
@click.option(
	'--use-gpu/--use-cpu',
	default=True,
	help='Whether to use GPU if available (default: True)',
)
@click.pass_context
def train_model(
	ctx,
	data_dir,
	out_dir,
	epochs,
	lr,
	batch_size,
	model_name,
	save_checkpoints,
	use_gpu,
):
	"""Train an MNIST classifier model.

	This command trains a convolutional neural network for MNIST digit classification.
	Training metrics and model checkpoints are saved to the specified output directory.
	"""
	logger = ctx.obj['logger']
	logger.info(f'Starting training with {epochs} epochs and batch size {batch_size}')

	# Generate a timestamp-based model name if not provided
	if model_name is None:
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		model_name = f'mnist_classifier_{timestamp}'

	# Create metadata for the training run
	metadata = {
		'model_name': model_name,
		'training_params': {
			'epochs': epochs,
			'learning_rate': lr,
			'batch_size': batch_size,
			'timestamp': datetime.now().isoformat(),
			'use_gpu': use_gpu,
		},
	}

	start_time = time.time()

	try:
		# Run the training
		metrics = train(
			data_dir=data_dir,
			out_dir=out_dir,
			model_name=model_name,
			num_epochs=epochs,
			learning_rate=lr,
			batch_size=batch_size,
			save_checkpoints=save_checkpoints,
			use_gpu=use_gpu,
		)

		training_time = time.time() - start_time

		# Update metadata with results
		metadata['training_time'] = f'{training_time:.2f} seconds'
		metadata['metrics'] = metrics

		# Save metadata
		out_dir.mkdir(parents=True, exist_ok=True)
		with open(out_dir / f'{model_name}_metadata.json', 'w') as f:
			json.dump(metadata, f, indent=2)

		logger.info(f'Training completed in {training_time:.2f} seconds')
		logger.info(f'Model saved as {model_name}')
		logger.info(f'Final accuracy: {metrics["final_val_accuracy"]:.4f}')

	except Exception as e:
		logger.exception(f'Training failed: {str(e)}')  # noqa: TRY401
		sys.exit(1)


@cli.command()
@click.option(
	'--model-path',
	type=click.Path(exists=True, dir_okay=False, path_type=Path),
	required=True,
	help='Path to the trained model file',
)
@click.option(
	'--image-path',
	type=click.Path(exists=True, dir_okay=False, path_type=Path),
	required=True,
	help='Path to the image for inference',
)
@click.option(
	'--output-path',
	type=click.Path(dir_okay=False, path_type=Path),
	default=None,
	help='Path to save the inference result (default: ./inference_results_{timestamp}.png)',
)
@click.option(
	'--use-gpu/--use-cpu',
	default=True,
	help='Whether to use GPU if available (default: True)',
)
@click.pass_context
def predict(ctx, model_path, image_path, output_path, use_gpu):
	"""Run inference with a trained model on a single image.

	This command uses a trained MNIST classifier to predict the digit in an image.
	The result is displayed with a confidence score and saved as a visualization.
	"""
	logger = ctx.obj['logger']

	if output_path is None:
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		output_path = Path(f'inference_result_{timestamp}.png')

	logger.info(f'Running inference with model: {model_path}')
	logger.info(f'Input image: {image_path}')

	try:
		# Run inference
		prediction, confidence = run_inference(
			model_path, image_path, output_path, use_gpu=use_gpu
		)

		# Print results in a user-friendly format
		click.echo('\nPrediction result:')
		click.echo(f'  Digit: {prediction}')
		click.echo(f'  Confidence: {confidence:.4f}')
		click.echo(f'  Visualization saved to: {output_path}')

	except Exception as e:
		logger.error(f'Inference failed: {str(e)}')
		if ctx.obj['verbose']:
			logger.exception('Detailed error information:')
		sys.exit(1)


@cli.group()
def dataset():
	"""Dataset management commands."""
	pass


@dataset.command()
@click.option(
	'--data-dir',
	type=click.Path(file_okay=False, path_type=Path),
	default=Path('data/rawdata'),
	help='Directory to download the MNIST dataset',
)
@click.option(
	'--samples-dir',
	type=click.Path(file_okay=False, path_type=Path),
	default=Path('data/samples'),
	help='Directory to save sample images',
)
@click.option(
	'--num-samples',
	type=click.IntRange(min=1),
	default=10,
	help='Number of sample images to generate',
)
@click.pass_context
def download(ctx, data_dir, samples_dir, num_samples):
	"""Download the MNIST dataset and generate sample images.

	This command downloads the MNIST dataset and saves sample images
	for visualization.
	"""
	from simple_ml_project.utils import download_and_prepare_dataset

	logger = ctx.obj['logger']
	logger.info(f'Downloading MNIST dataset to {data_dir}')

	try:
		download_and_prepare_dataset(
			data_dir=data_dir, samples_dir=samples_dir, num_samples=num_samples
		)
		logger.info('Dataset downloaded successfully')
		logger.info(f'Sample images saved to {samples_dir}')

	except Exception as e:
		logger.error(f'Dataset download failed: {str(e)}')
		if ctx.obj['verbose']:
			logger.exception('Detailed error information:')
		sys.exit(1)


if __name__ == '__main__':
	cli(obj={})

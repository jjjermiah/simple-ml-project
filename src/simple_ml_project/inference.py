# src/simple_ml_project/inference.py

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torchvision import transforms

from simple_ml_project.utils import console, load_model, logger


def load_image(
	path: Path, transform: Optional[transforms.Compose] = None
) -> torch.Tensor:
	"""
	Load and preprocess an image for MNIST classification.

	Args:
	    path: Path to the image file
	    transform: Optional custom transform to apply

	Returns:
	    Preprocessed tensor of shape (1, 1, 28, 28)
	"""
	try:
		# Open image, convert to grayscale, and resize to 28x28
		img = Image.open(path).convert('L').resize((28, 28))
	except Exception as e:
		logger.exception(f'Failed to open or process image {path}: {str(e)}')
		raise

	# Use default transform if none provided
	if transform is None:
		transform = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
			]
		)

	# Add batch dimension
	tensor = transform(img).unsqueeze(0)
	return tensor


def get_prediction(
	model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device
) -> Tuple[int, float, List[float]]:
	"""
	Get prediction from model for a single image.

	Args:
	    model: Trained model
	    image_tensor: Input image tensor of shape (1, 1, 28, 28)
	    device: Device to run inference on

	Returns:
	    Tuple of (predicted_class, confidence, all_probabilities)
	"""
	# Ensure model is in evaluation mode
	model.eval()

	# Move input to device
	image_tensor = image_tensor.to(device)

	# Run inference
	with torch.no_grad():
		# Get logits
		logits = model(image_tensor)

		# Get probabilities
		probabilities = F.softmax(logits, dim=1)[0]

		# Get predicted class and confidence
		pred_class = torch.argmax(probabilities).item()
		confidence = probabilities[pred_class].item()

	return pred_class, confidence, probabilities.cpu().tolist()


def run_inference(
	model_path: Path,
	image_path: Path,
	output_path: Optional[Path] = None,
	use_gpu: bool = True,
) -> Tuple[int, float]:
	"""
	Run inference with a trained model on a single image.

	Args:
	    model_path: Path to the trained model file
	    image_path: Path to the image for inference
	    output_path: Path to save the visualization result
	    use_gpu: Whether to use GPU if available

	Returns:
	    Tuple of (predicted_class, confidence)
	"""
	# Default output path if not specified
	if output_path is None:
		output_path = Path('inference_result.png')

	# Set device
	device_name = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
	device = torch.device(device_name)
	console.print(f'[bold blue]Using device:[/] [cyan]{device}[/] for inference')

	# Load the model
	try:
		model, metadata = load_model(model_path, device=device)
		console.print(f'[bold green]✓[/] Loaded model from [cyan]{model_path}[/]')
		if metadata:
			console.print(
				f'[bold blue]Model metadata:[/] val_accuracy=[cyan]{metadata.get("val_accuracy", "N/A")}[/]'
			)
	except Exception as e:
		logger.exception(f'Failed to load model from {model_path}: {str(e)}')
		raise

	# Load and preprocess the image
	try:
		img_tensor = load_image(image_path)
		console.print(
			f'[bold green]✓[/] Loaded and preprocessed image from [cyan]{image_path}[/]'
		)
	except Exception as e:
		logger.exception(f'Failed to load image from {image_path}: {str(e)}')
		raise

	# Get prediction
	try:
		pred_class, confidence, probabilities = get_prediction(
			model, img_tensor, device
		)
		console.print(
			f'[bold green]✓[/] Predicted class: [cyan]{pred_class}[/] with confidence: [cyan]{confidence:.4f}[/]'
		)
	except Exception as e:
		logger.exception(f'Prediction failed: {str(e)}')
		raise

	# Create visualization
	try:
		create_prediction_visualization(
			img_tensor=img_tensor,
			probabilities=probabilities,
			pred_class=pred_class,
			confidence=confidence,
			output_path=output_path,
		)
		console.print(
			f'[bold green]✓[/] Saved inference visualization to [cyan]{output_path}[/]'
		)
	except Exception as e:
		logger.exception(f'Failed to create visualization: {str(e)}')
		# Continue with returning prediction even if visualization fails

	return pred_class, confidence


def create_prediction_visualization(
	img_tensor: torch.Tensor,
	probabilities: List[float],
	pred_class: int,
	confidence: float,
	output_path: Path,
) -> None:
	"""
	Create and save a visualization of the prediction.

	Args:
	    img_tensor: Input image tensor
	    probabilities: List of prediction probabilities for all classes
	    pred_class: Predicted class index
	    confidence: Confidence score for the prediction
	    output_path: Path to save the visualization
	"""
	# Create figure
	plt.figure(figsize=(12, 6))

	# Show the input image
	plt.subplot(1, 2, 1)
	plt.imshow(img_tensor[0, 0].numpy(), cmap='gray')
	plt.title('Input Image')
	plt.axis('off')

	# Show the class probabilities
	plt.subplot(1, 2, 2)
	bars = plt.bar(range(10), probabilities, tick_label=list(range(10)))

	# Highlight the predicted class
	bars[pred_class].set_color('red')

	plt.xlabel('Digit Class')
	plt.ylabel('Probability')
	plt.title(f'Predicted: {pred_class} (Confidence: {confidence:.2f})')
	plt.ylim(0, 1)

	# Save the figure
	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path)

	# Close the plot to free memory
	plt.close()


def batch_inference(
	model_path: Path,
	image_paths: List[Path],
	output_dir: Path,
	use_gpu: bool = True,
) -> Dict[str, Dict[str, Union[int, float]]]:
	"""
	Run inference on multiple images with the same model.

	Args:
	    model_path: Path to the trained model file
	    image_paths: List of paths to images for inference
	    output_dir: Directory to save visualization results
	    use_gpu: Whether to use GPU if available

	Returns:
	    Dictionary mapping image paths to prediction results
	"""
	# Set device
	device_name = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
	device = torch.device(device_name)
	console.print(f'[bold blue]Using device:[/] [cyan]{device}[/] for batch inference')

	# Create output directory
	output_dir.mkdir(parents=True, exist_ok=True)

	# Load model once for all images
	model, _ = load_model(model_path, device=device)
	console.print(f'[bold green]✓[/] Loaded model from [cyan]{model_path}[/]')

	# Process each image
	results = {}
	for image_path in image_paths:
		try:
			# Generate output path
			output_path = output_dir / f'{image_path.stem}_result.png'

			# Load image
			img_tensor = load_image(image_path)

			# Get prediction
			pred_class, confidence, probabilities = get_prediction(
				model, img_tensor, device
			)

			# Create visualization
			create_prediction_visualization(
				img_tensor=img_tensor,
				probabilities=probabilities,
				pred_class=pred_class,
				confidence=confidence,
				output_path=output_path,
			)

			# Store result
			results[str(image_path)] = {
				'predicted_class': pred_class,
				'confidence': confidence,
				'probabilities': probabilities,
				'result_image': str(output_path),
			}

			logger.info(
				f'Processed {image_path}: class={pred_class}, confidence={confidence:.4f}'
			)

		except Exception as e:
			logger.error(f'Failed to process {image_path}: {str(e)}')
			results[str(image_path)] = {'error': str(e)}

	return results


def main() -> None:
	"""Legacy entry point for backward compatibility."""
	parser = argparse.ArgumentParser(description='Run inference with a trained model')
	parser.add_argument(
		'--model_path',
		type=Path,
		default=Path('data/results/models/mnist_classifier_final.pt'),
		help='Path to the model file',
	)
	parser.add_argument(
		'--image_path', type=Path, required=True, help='Path to the image file'
	)
	parser.add_argument(
		'--output_path',
		type=Path,
		default=Path('inference_result.png'),
		help='Path to save the inference result',
	)
	parser.add_argument('--no_gpu', action='store_true', help='Disable GPU usage')

	args = parser.parse_args()

	# Run inference
	pred_class, confidence = run_inference(
		args.model_path, args.image_path, args.output_path, use_gpu=not args.no_gpu
	)

	print(f'Predicted class: {pred_class}')
	print(f'Confidence: {confidence:.4f}')
	print(f'Visualization saved to: {args.output_path}')


if __name__ == '__main__':
	main()

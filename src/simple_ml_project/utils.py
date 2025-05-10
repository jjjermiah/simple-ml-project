# src/simple_ml_project/utils.py

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
	BarColumn,
	Progress,
	TaskProgressColumn,
	TextColumn,
	TimeRemainingColumn,
)
from torch import nn
from torchvision import datasets, transforms

# Set up Rich console
console = Console()

# Configure Rich logging
logging.basicConfig(
	level=logging.INFO,
	format='%(message)s',
	datefmt='[%X]',
	handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)


def download_and_prepare_dataset(
	data_dir: Path,
	samples_dir: Path,
	num_samples: int = 10,
) -> None:
	"""
	Download the MNIST dataset and save sample images.

	Args:
	    data_dir: Directory to download the dataset
	    samples_dir: Directory to save sample images
	    num_samples: Number of sample images to generate
	"""
	# Create directories
	data_dir.mkdir(parents=True, exist_ok=True)
	samples_dir.mkdir(parents=True, exist_ok=True)

	# Data transformation
	transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
		]
	)

	# Download and load the dataset
	console.print('[bold blue]Downloading MNIST dataset...[/]')
	with console.status(
		'[bold green]Downloading training dataset...', spinner='dots'
	) as status:
		train_dataset = datasets.MNIST(
			root=data_dir, train=True, download=True, transform=transform
		)
		status.update('[bold green]Downloading test dataset...')
		test_dataset = datasets.MNIST(
			root=data_dir, train=False, download=True, transform=transform
		)

	console.print('[bold green]✓[/] Dataset download complete:')
	console.print(f'  - Training set: [cyan]{len(train_dataset)}[/] samples')
	console.print(f'  - Test set: [cyan]{len(test_dataset)}[/] samples')

	# Save sample images
	console.print(f'[bold blue]Saving {num_samples} sample images to {samples_dir}[/]')

	# Use an inverse transform to convert tensors back to images
	inv_normalize = transforms.Compose(
		[
			transforms.Normalize((-0.1307 / 0.3081,), (1 / 0.3081,)),
		]
	)

	# Create a grid of sample images
	rows = int(np.ceil(num_samples / 5))
	cols = min(5, num_samples)

	fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
	axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

	# Use Rich progress bar for saving samples
	with Progress(
		TextColumn('[bold blue]{task.description}'),
		BarColumn(),
		TaskProgressColumn(),
		TimeRemainingColumn(),
		console=console,
	) as progress:
		task = progress.add_task(
			'Saving samples', total=min(num_samples, len(train_dataset))
		)

		for i in range(min(num_samples, len(train_dataset))):
			img, label = train_dataset[i]

			# Denormalize for visualization
			img_viz = inv_normalize(img).squeeze().numpy()

			# Plot in the grid
			axes[i].imshow(img_viz, cmap='gray')
			axes[i].set_title(f'Label: {label}')
			axes[i].axis('off')

			# Also save individual images
			plt.imsave(
				samples_dir / f'digit_{label}_sample_{i}.png', img_viz, cmap='gray'
			)

			progress.update(task, advance=1)

	# Save the grid
	plt.tight_layout()
	plt.savefig(samples_dir / 'mnist_samples_grid.png')
	plt.close()

	# Create a dataset info file
	dataset_info = {
		'name': 'MNIST',
		'train_samples': len(train_dataset),
		'test_samples': len(test_dataset),
		'image_shape': '28x28',
		'num_classes': 10,
		'classes': list(range(10)),
		'download_date': datetime.now().isoformat(),
	}

	with open(samples_dir / 'dataset_info.json', 'w') as f:
		json.dump(dataset_info, f, indent=2)

	console.print('[bold green]✓[/] Dataset preparation complete')


def save_model(
	model: nn.Module,
	path: Path,
	metadata: Optional[Dict[str, Any]] = None,
	optimizer: Optional[torch.optim.Optimizer] = None,
	epoch: Optional[int] = None,
) -> None:
	"""
	Save a model with metadata.

	Args:
	    model: Model to save
	    path: Path to save the model
	    metadata: Optional metadata to save with the model
	    optimizer: Optional optimizer state to save
	    epoch: Optional current epoch number
	"""
	# Create save directory if it doesn't exist
	path.parent.mkdir(parents=True, exist_ok=True)

	# Prepare the save dictionary
	save_dict = {
		'model_state_dict': model.state_dict(),
		'model_class': model.__class__.__name__,
	}

	# Add optional components
	if metadata is not None:
		save_dict['metadata'] = metadata

	if optimizer is not None:
		save_dict['optimizer_state_dict'] = optimizer.state_dict()

	if epoch is not None:
		save_dict['epoch'] = epoch

	# Add timestamp
	save_dict['timestamp'] = datetime.now().isoformat()

	# Save the model with a status display
	with console.status(
		f'[bold yellow]Saving model to {path}...', spinner='dots'
	) as status:
		torch.save(save_dict, path)

	# Print success message
	console.print(f'[bold green]✓[/] Model saved to [cyan]{path}[/]')

	# If we have metadata, print some key metrics
	if metadata and isinstance(metadata, dict):
		console.print('[bold blue]Model metrics:[/]')
		table_data = []

		for key, value in metadata.items():
			if key in [
				'val_accuracy',
				'train_accuracy',
				'val_loss',
				'train_loss',
				'best_val_accuracy',
			]:
				if isinstance(value, (int, float)):
					console.print(f'  - {key}: [cyan]{value:.4f}[/]')


def load_model(
	path: Path,
	model_class: Optional[nn.Module] = None,
	device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
	"""
	Load a model with metadata.

	Args:
	    path: Path to the saved model
	    model_class: Optional model class to use
	    device: Device to load the model to

	Returns:
	    Tuple of (model, metadata)
	"""
	# Set device if not provided
	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load the save dictionary with status display
	with console.status(
		f'[bold yellow]Loading model from {path}...', spinner='dots'
	) as status:
		checkpoint = torch.load(path, map_location=device, weights_only=False)

	# Extract model state and metadata
	model_state = checkpoint.get('model_state_dict')
	metadata = checkpoint.get('metadata', {})

	# If model_class is not provided, try to determine it from the checkpoint
	if model_class is None:
		status_text = '[bold yellow]Initializing model...'
		with console.status(status_text, spinner='dots'):
			from simple_ml_project.model import MNISTClassifier

			model_class = MNISTClassifier

	# Create and load model
	with console.status('[bold yellow]Applying model weights...', spinner='dots'):
		model = model_class().to(device)
		model.load_state_dict(model_state)
		model.eval()

	# Print success message
	console.print(f'[bold green]✓[/] Model loaded from [cyan]{path}[/]')

	# Print metadata if available
	if metadata:
		console.print('[bold blue]Model metadata:[/]')
		for key, value in metadata.items():
			if key in [
				'val_accuracy',
				'train_accuracy',
				'val_loss',
				'train_loss',
				'best_val_accuracy',
			] and isinstance(value, (int, float)):
				console.print(f'  - {key}: [cyan]{value:.4f}[/]')

	return model, metadata


def create_confusion_matrix(
	predictions: List[int],
	labels: List[int],
	num_classes: int = 10,
	save_path: Optional[Path] = None,
) -> np.ndarray:
	"""
	Create a confusion matrix from predictions and labels.

	Args:
	    predictions: List of predicted classes
	    labels: List of true labels
	    num_classes: Number of classes
	    save_path: Optional path to save the confusion matrix visualization

	Returns:
	    Confusion matrix as a numpy array
	"""
	# Create confusion matrix with progress display
	with console.status('[bold yellow]Creating confusion matrix...', spinner='dots'):
		conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

		for pred, label in zip(predictions, labels, strict=False):
			conf_matrix[label, pred] += 1

	# Print some basic metrics from the confusion matrix
	correct = sum(conf_matrix[i, i] for i in range(num_classes))
	total = sum(sum(row) for row in conf_matrix)
	accuracy = correct / total if total > 0 else 0

	console.print('[bold blue]Confusion Matrix Stats:[/]')
	console.print(f'  - Overall Accuracy: [cyan]{accuracy:.4f}[/] ({correct}/{total})')

	# Calculate per-class precision and recall
	for i in range(num_classes):
		class_total = sum(conf_matrix[i, :])
		class_correct = conf_matrix[i, i]
		class_precision = (
			class_correct / sum(conf_matrix[:, i]) if sum(conf_matrix[:, i]) > 0 else 0
		)
		class_recall = class_correct / class_total if class_total > 0 else 0

		console.print(
			f'  - Class {i}: Precision: [cyan]{class_precision:.4f}[/], Recall: [cyan]{class_recall:.4f}[/]'
		)

	# If save_path is provided, create and save visualization
	if save_path is not None:
		with console.status(
			f'[bold yellow]Saving confusion matrix to {save_path}...', spinner='dots'
		):
			plt.figure(figsize=(10, 8))
			plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
			plt.title('Confusion Matrix')
			plt.colorbar()

			# Add labels
			tick_marks = np.arange(num_classes)
			plt.xticks(tick_marks, range(num_classes))
			plt.yticks(tick_marks, range(num_classes))
			plt.xlabel('Predicted Label')
			plt.ylabel('True Label')

			# Add counts
			thresh = conf_matrix.max() / 2
			for i in range(num_classes):
				for j in range(num_classes):
					plt.text(
						j,
						i,
						str(conf_matrix[i, j]),
						horizontalalignment='center',
						color='white' if conf_matrix[i, j] > thresh else 'black',
					)

			plt.tight_layout()
			save_path.parent.mkdir(parents=True, exist_ok=True)
			plt.savefig(save_path)
			plt.close()

		console.print(
			f'[bold green]✓[/] Confusion matrix saved to [cyan]{save_path}[/]'
		)

	return conf_matrix

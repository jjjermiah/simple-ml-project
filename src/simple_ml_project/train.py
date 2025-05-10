# src/simple_ml_project/train.py

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.logging import RichHandler
from rich.progress import (
	BarColumn,
	Progress,
	TaskProgressColumn,
	TextColumn,
	TimeRemainingColumn,
)
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from simple_ml_project.model import MNISTClassifier
from simple_ml_project.utils import console, create_confusion_matrix, save_model

# Set up logging with Rich
logging.basicConfig(
	level=logging.INFO,
	format='%(message)s',
	datefmt='[%X]',
	handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def get_dataloaders(
	data_dir: Path,
	batch_size: int,
	val_split: float = 0.1,
	save_samples: bool = True,
	num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
	"""
	Create training and validation dataloaders for MNIST.

	Args:
	    data_dir: Directory to store/load the MNIST dataset
	    batch_size: Batch size for training
	    val_split: Fraction of training data to use for validation
	    save_samples: Whether to save sample images
	    num_workers: Number of worker processes for data loading

	Returns:
	    Tuple of (train_dataloader, validation_dataloader)
	"""
	# Data transformation
	transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
		]
	)

	# Download and load the dataset
	try:
		full_dataset = datasets.MNIST(
			root=data_dir, train=True, download=True, transform=transform
		)
	except Exception as e:
		logger.error(f'Failed to download or load MNIST dataset: {str(e)}')
		raise

	# Save sample images if requested
	if save_samples:
		logger.info('Saving sample images to data directory')
		samples_dir = data_dir.parent / 'samples'
		samples_dir.mkdir(exist_ok=True, parents=True)

		fig, axes = plt.subplots(2, 5, figsize=(12, 5))
		for i, ax in enumerate(axes.flatten()):
			img, label = full_dataset[i]
			ax.imshow(img.squeeze().numpy(), cmap='gray')
			ax.set_title(f'Label: {label}')
			ax.axis('off')

			# Also save individual images
			plt.imsave(
				samples_dir / f'sample_{i}_label_{label}.png',
				img.squeeze().numpy(),
				cmap='gray',
			)

		# Save the figure with all samples
		plt.tight_layout()
		plt.savefig(samples_dir / 'mnist_samples.png')
		plt.close()

	# Split into training and validation sets
	val_size = int(len(full_dataset) * val_split)
	train_size = len(full_dataset) - val_size

	# Set generator for reproducibility
	generator = torch.Generator().manual_seed(42)
	train_dataset, val_dataset = random_split(
		full_dataset, [train_size, val_size], generator=generator
	)

	# Create data loaders
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
	)

	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)

	logger.info(
		f'Created dataloaders with {train_size} training samples and {val_size} validation samples'
	)
	return train_loader, val_loader


def evaluate_model(
	model: nn.Module,
	dataloader: DataLoader,
	device: torch.device,
	criterion: nn.Module = None,
) -> Dict[str, Any]:
	"""
	Evaluate the model on the provided dataloader.

	Args:
	    model: The model to evaluate
	    dataloader: Dataloader with evaluation data
	    device: Device to run evaluation on
	    criterion: Loss function to use (defaults to CrossEntropyLoss if None)

	Returns:
	    Dictionary with evaluation metrics
	"""
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0

	if criterion is None:
		criterion = nn.CrossEntropyLoss()

	all_preds = []
	all_labels = []

	# Use Rich progress bar for evaluation
	with Progress(
		TextColumn('[bold green]{task.description}'),
		BarColumn(),
		TaskProgressColumn(),
		TimeRemainingColumn(),
		console=console,
	) as progress:
		eval_task = progress.add_task('[Evaluation]', total=len(dataloader))

		with torch.no_grad():
			for inputs, labels in dataloader:
				inputs, labels = inputs.to(device), labels.to(device)

				# Forward pass
				outputs = model(inputs)
				loss = criterion(outputs, labels)

				# Calculate accuracy
				_, predicted = torch.max(outputs, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				total_loss += loss.item() * inputs.size(0)  # Weight by batch size

				# Store predictions and labels for confusion matrix
				all_preds.extend(predicted.cpu().numpy())
				all_labels.extend(labels.cpu().numpy())

				# Update progress bar
				progress.update(eval_task, advance=1)

	# Calculate metrics
	accuracy = correct / total
	avg_loss = total_loss / total

	# Create per-class metrics
	class_correct = [0] * 10
	class_total = [0] * 10

	# Count per-class correct predictions
	for pred, label in zip(all_preds, all_labels, strict=False):
		class_correct[label] += pred == label
		class_total[label] += 1

	# Calculate per-class accuracy
	class_accuracy = [
		class_correct[i] / class_total[i] if class_total[i] > 0 else 0
		for i in range(10)
	]

	return {
		'loss': avg_loss,
		'accuracy': accuracy,
		'predictions': all_preds,
		'labels': all_labels,
		'class_accuracy': class_accuracy,
	}


def train(
	data_dir: Path,
	out_dir: Path,
	model_name: str = None,
	num_epochs: int = 5,
	learning_rate: float = 1e-3,
	batch_size: int = 256,
	save_checkpoints: bool = True,
	use_gpu: bool = True,
) -> Dict[str, Any]:
	"""
	Train an MNIST classifier.

	Args:
	    data_dir: Directory containing input data
	    out_dir: Directory to save model and results
	    model_name: Name for the saved model
	    num_epochs: Number of training epochs
	    learning_rate: Learning rate
	    batch_size: Batch size for training
	    save_checkpoints: Whether to save intermediate model checkpoints
	    use_gpu: Whether to use GPU if available

	Returns:
	    Dictionary with training metrics
	"""
	# Create output directory
	out_dir.mkdir(parents=True, exist_ok=True)
	models_dir = out_dir / 'models'
	models_dir.mkdir(exist_ok=True, parents=True)

	# Generate model name if not provided
	if model_name is None:
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		model_name = f'mnist_classifier_{timestamp}'

	# Set up device
	device_name = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
	device = torch.device(device_name)
	logger.info(f'Using device: {device}')

	# Create model
	model = MNISTClassifier().to(device)
	logger.info('Created MNIST classifier model')

	# Loss function and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode='min',
		factor=0.5,
		patience=2,
	)

	# Get data loaders
	train_loader, val_loader = get_dataloaders(data_dir, batch_size)

	# Initialize metrics tracking
	train_losses = []
	val_losses = []
	train_accuracies = []
	val_accuracies = []
	best_val_accuracy = 0.0
	best_confusion_matrix = None

	# Training loop
	console.print(f'[bold blue]Starting training for {num_epochs} epochs[/]')
	start_time = time.time()

	for epoch in range(num_epochs):
		epoch_start_time = time.time()

		# Training phase
		model.train()
		train_loss = 0.0
		train_correct = 0
		train_total = 0

		# Use Rich progress bar
		with Progress(
			TextColumn('[bold blue]{task.description}'),
			BarColumn(),
			TaskProgressColumn(),
			TextColumn('[bold]{task.fields[loss]:.4f}'),
			TextColumn('[cyan]{task.fields[accuracy]:.4f}'),
			TimeRemainingColumn(),
			console=console,
		) as progress:
			# Add task for training
			train_task = progress.add_task(
				f'Epoch {epoch + 1}/{num_epochs} [Train]',
				total=len(train_loader),
				loss=0.0,
				accuracy=0.0,
			)

			for inputs, labels in train_loader:
				inputs, labels = inputs.to(device), labels.to(device)

				# Zero the gradients
				optimizer.zero_grad()

				# Forward pass
				outputs = model(inputs)
				loss = criterion(outputs, labels)

				# Backward pass and optimize
				loss.backward()
				optimizer.step()

				# Update metrics
				train_loss += loss.item() * inputs.size(0)
				_, predicted = torch.max(outputs, 1)
				train_total += labels.size(0)
				train_correct += (predicted == labels).sum().item()

				# Update progress bar
				current_accuracy = train_correct / train_total
				progress.update(
					train_task, advance=1, loss=loss.item(), accuracy=current_accuracy
				)

		# Calculate epoch metrics
		epoch_train_loss = train_loss / train_total
		epoch_train_accuracy = train_correct / train_total
		train_losses.append(epoch_train_loss)
		train_accuracies.append(epoch_train_accuracy)

		# Validation phase
		val_metrics = evaluate_model(model, val_loader, device, criterion)
		val_losses.append(val_metrics['loss'])
		val_accuracies.append(val_metrics['accuracy'])

		# Update learning rate scheduler
		scheduler.step(val_metrics['loss'])

		# Log progress
		epoch_time = time.time() - epoch_start_time
		logger.info(
			f'Epoch {epoch + 1}/{num_epochs} - '
			f'Train Loss: {epoch_train_loss:.4f}, '
			f'Train Acc: {epoch_train_accuracy:.4f}, '
			f'Val Loss: {val_metrics["loss"]:.4f}, '
			f'Val Acc: {val_metrics["accuracy"]:.4f}, '
			f'Time: {epoch_time:.2f}s'
		)

		# Log per-class accuracy
		class_acc = val_metrics['class_accuracy']
		logger.info(
			'Per-class accuracy: '
			+ ' '.join([f'Class {i}: {acc:.4f}' for i, acc in enumerate(class_acc)])
		)

		# Save checkpoint if improved
		if val_metrics['accuracy'] > best_val_accuracy:
			best_val_accuracy = val_metrics['accuracy']
			logger.info(f'New best validation accuracy: {best_val_accuracy:.4f}')

			# Create confusion matrix
			predictions = val_metrics['predictions']
			labels = val_metrics['labels']
			best_confusion_matrix = create_confusion_matrix(
				predictions,
				labels,
				save_path=out_dir / f'{model_name}_confusion_matrix.png',
			)

			# Save the best model with metadata
			model_metadata = {
				'val_accuracy': best_val_accuracy,
				'val_loss': val_metrics['loss'],
				'train_accuracy': epoch_train_accuracy,
				'train_loss': epoch_train_loss,
				'epoch': epoch,
				'class_accuracy': class_acc,
			}

			save_model(
				model=model,
				path=models_dir / f'{model_name}_best.pt',
				metadata=model_metadata,
				optimizer=optimizer,
				epoch=epoch,
			)

		# Save intermediate checkpoints if requested
		if save_checkpoints and (epoch + 1) % 5 == 0:
			checkpoint_metadata = {
				'val_accuracy': val_metrics['accuracy'],
				'val_loss': val_metrics['loss'],
				'train_accuracy': epoch_train_accuracy,
				'train_loss': epoch_train_loss,
				'class_accuracy': class_acc,
			}

			save_model(
				model=model,
				path=models_dir / f'{model_name}_epoch_{epoch + 1}.pt',
				metadata=checkpoint_metadata,
				optimizer=optimizer,
				epoch=epoch,
			)

	# Training complete
	total_time = time.time() - start_time
	logger.info(f'Training completed in {total_time:.2f} seconds')

	# Save final model
	final_metadata = {
		'val_accuracy': val_accuracies[-1],
		'val_loss': val_losses[-1],
		'train_accuracy': train_accuracies[-1],
		'train_loss': train_losses[-1],
		'best_val_accuracy': best_val_accuracy,
		'training_time': total_time,
		'class_accuracy': val_metrics['class_accuracy'],
	}

	save_model(
		model=model,
		path=models_dir / f'{model_name}_final.pt',
		metadata=final_metadata,
		optimizer=optimizer,
		epoch=num_epochs - 1,
	)
	logger.info(f'Saved final model to {models_dir / f"{model_name}_final.pt"}')

	# Create and save training plots
	plt.figure(figsize=(12, 5))

	# Loss plot
	plt.subplot(1, 2, 1)
	plt.plot(train_losses, label='Train Loss')
	plt.plot(val_losses, label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Training and Validation Loss')

	# Accuracy plot
	plt.subplot(1, 2, 2)
	plt.plot(train_accuracies, label='Train Accuracy')
	plt.plot(val_accuracies, label='Validation Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.title('Training and Validation Accuracy')

	# Save the plot
	plt.tight_layout()
	plt.savefig(out_dir / f'{model_name}_training_plots.png')

	# Return training metrics
	metrics = {
		'train_losses': train_losses,
		'val_losses': val_losses,
		'train_accuracies': train_accuracies,
		'val_accuracies': val_accuracies,
		'best_val_accuracy': best_val_accuracy,
		'training_time': total_time,
		'final_train_accuracy': train_accuracies[-1],
		'final_val_accuracy': val_accuracies[-1],
		'class_accuracy': val_metrics['class_accuracy'],
	}

	# Save metrics to a JSON file
	with open(out_dir / f'{model_name}_metrics.json', 'w') as f:
		# Convert numpy arrays to lists for JSON serialization
		json_metrics = {
			k: v
			if not isinstance(v, (np.ndarray, list)) or k in ['class_accuracy']
			else [float(x) for x in v]
			for k, v in metrics.items()
		}
		json.dump(json_metrics, f, indent=2)

	# Display a Rich summary panel for the completed training
	console.rule('[bold green]Training Complete')
	console.print(f'[bold blue]Model:[/] {model_name}')
	console.print(f'[bold blue]Training time:[/] {total_time:.2f} seconds')
	console.print('[bold blue]Final metrics:[/]')
	console.print(f'  - Train accuracy: [cyan]{train_accuracies[-1]:.4f}[/]')
	console.print(f'  - Validation accuracy: [cyan]{val_accuracies[-1]:.4f}[/]')
	console.print(f'  - Train loss: [cyan]{train_losses[-1]:.4f}[/]')
	console.print(f'  - Validation loss: [cyan]{val_losses[-1]:.4f}[/]')
	console.print(
		f'[bold blue]Best validation accuracy:[/] [cyan]{best_val_accuracy:.4f}[/]'
	)
	console.print(f'[bold blue]Model saved to:[/] [cyan]{out_dir / model_name}[/]')
	console.rule()

	return metrics


def main() -> None:
	"""Legacy entry point for backward compatibility."""
	parser = argparse.ArgumentParser(description='Train an MNIST classifier')
	parser.add_argument('--data_dir', type=Path, default=Path('data/rawdata'))
	parser.add_argument('--out_dir', type=Path, default=Path('data/results'))
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--model_name', type=str, default=None)
	parser.add_argument(
		'--save_checkpoints', action='store_true', help='Save intermediate checkpoints'
	)
	parser.add_argument('--no_gpu', action='store_true', help='Disable GPU usage')

	args = parser.parse_args()

	# Call the train function with the parsed arguments
	train(
		data_dir=args.data_dir,
		out_dir=args.out_dir,
		model_name=args.model_name,
		num_epochs=args.epochs,
		learning_rate=args.lr,
		batch_size=args.batch_size,
		save_checkpoints=args.save_checkpoints,
		use_gpu=not args.no_gpu,
	)


if __name__ == '__main__':
	main()

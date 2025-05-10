# src/simple_ml_project/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Optional, List, Union


class MNISTClassifier(nn.Module):
    """MNIST classifier model with convolutional layers."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25) -> None:
        """
        Initialize the MNIST classifier.
        
        Args:
            num_classes: Number of output classes (default: 10 for digits 0-9)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),
        )
        
        # Calculate the flattened feature size (7x7 due to input 28x28 with two 2x2 max pools)
        self.fc_size = 64 * 7 * 7
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 2),  # Higher dropout for fully connected layers
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def predict(self, x: torch.Tensor) -> Tuple[int, float]:
        """Get the predicted class and confidence.
        
        Args:
            x: Input tensor of shape (1, 1, 28, 28)
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        with torch.no_grad():
            logits = self(x)
            probabilities = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item()
            
        return pred_class, confidence
    
    def predict_batch(self, x: torch.Tensor) -> Tuple[List[int], List[float]]:
        """Get predicted classes and confidences for a batch of images.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Tuple of (predicted_classes, confidences)
        """
        with torch.no_grad():
            logits = self(x)
            probabilities = F.softmax(logits, dim=1)
            pred_classes = torch.argmax(probabilities, dim=1).cpu().tolist()
            confidences = [probabilities[i, cls].item() for i, cls in enumerate(pred_classes)]
            
        return pred_classes, confidences
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings from the model.
        
        This can be useful for visualization, similarity search, or transfer learning.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Embedding tensor from the penultimate layer
        """
        with torch.no_grad():
            # Extract features from convolutional layers
            features = self.features(x)
            
            # Flatten
            features = features.view(features.size(0), -1)
            
            # Get embeddings from the first part of classifier (all except last layer)
            for i, layer in enumerate(self.classifier):
                if isinstance(layer, nn.Linear) and i == len(self.classifier) - 1:
                    break
                features = layer(features)
                
        return features


# For backward compatibility
UNet = MNISTClassifier  # Alias for compatibility with existing code
"""
MNIST Classification using Softmax

Objective:
- Classify handwritten digits from the MNIST database using a Softmax classifier.
"""

# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# GPU Monitoring
print("="*80)
print("GPU STATUS")
print("="*80)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
print("="*80)

# Set random seed for reproducibility
torch.manual_seed(0)

# Hyperparameters
batch_size = 100
learning_rate = 0.1
num_epochs = 10
input_size = 28 * 28  # 784 pixels
num_classes = 10  # digits 0-9

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform
)

# Data loaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Define the model
class SoftmaxClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = x.reshape(-1, input_size)
        return self.linear(x)

# Initialize the model
model = SoftmaxClassifier(input_size, num_classes)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training function
def train_model():
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')
                
            loss_list.append(loss.item())
            acc_list.append(correct / total)
    
    return loss_list, acc_list

# Test the model
def test_model():
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy

# Plot training curves
def plot_training_curves(loss_list, acc_list):
    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(acc_list, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Visualize some predictions
def visualize_predictions():
    # Get some random test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Move to device
    images = images.to(device)
    labels = labels.to(device)
    
    # Get predictions
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # Move images and labels back to CPU for plotting
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()
    
    # Plot the images with their predicted labels
    fig = plt.figure(figsize=(12, 8))
    for idx in range(6):
        ax = fig.add_subplot(2, 3, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx].squeeze(), cmap='gray')
        ax.set_title(f'Predicted: {predicted[idx].item()}\nActual: {labels[idx].item()}', 
                    color='green' if predicted[idx] == labels[idx] else 'red')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train the model
    print("Starting training...")
    loss_list, acc_list = train_model()
    
    # Test the model
    test_accuracy = test_model()
    
    # Plot training curves
    plot_training_curves(loss_list, acc_list)
    
    # Visualize some predictions
    visualize_predictions()
    
    # Print GPU memory usage if available
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

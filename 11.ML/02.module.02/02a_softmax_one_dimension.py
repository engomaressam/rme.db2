"""
Softmax Classifier 1D

Objective:
- Build a Softmax classifier using PyTorch's Sequential module for multi-class classification.
"""

# Import libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

# Create the dataset
class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-2, 2, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x > -1.0)[:, 0] * (self.x < 1.0)[:, 0]] = 1
        self.y[(self.x >= 1.0)[:, 0]] = 2
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

# Helper function to plot data and decision boundaries
def plot_data(data_set, model=None, title=None):
    X = data_set[:][0]
    Y = data_set[:][1]
    
    plt.figure(figsize=(10, 5))
    
    # Plot data points
    plt.plot(X[Y == 0].numpy(), np.zeros_like(X[Y == 0].numpy()), 'bo', label='Class 0')
    plt.plot(X[Y == 1].numpy(), np.ones_like(X[Y == 1].numpy()), 'ro', label='Class 1')
    plt.plot(X[Y == 2].numpy(), 2 * np.ones_like(X[Y == 2].numpy()), 'go', label='Class 2')
    
    # Plot model predictions if provided
    if model is not None:
        # Generate test points
        x_test = torch.linspace(-2, 2, 100).view(-1, 1)
        with torch.no_grad():
            y_test = model(x_test)
            _, predicted = torch.max(y_test, 1)
        
        # Plot decision boundaries
        for i in range(len(x_test)-1):
            if predicted[i] != predicted[i+1]:
                plt.axvline(x=x_test[i].item(), color='k', linestyle='--', alpha=0.5)
    
    plt.ylim(-0.5, 2.5)
    plt.xlabel('Input x')
    plt.ylabel('Class')
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def train_model(epochs=300, learning_rate=0.01, batch_size=5):
    # Create dataset and dataloader
    data_set = Data()
    train_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(1, 3)  # 1 input feature, 3 output classes
    )
    
    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for x, y in train_loader:
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    return model, data_set

if __name__ == "__main__":
    # Plot initial data
    data_set = Data()
    plot_data(data_set, title="Original Data Distribution")
    
    # Train the model
    model, data_set = train_model(epochs=300, learning_rate=0.01)
    
    # Plot final decision boundaries
    plot_data(data_set, model, title="Decision Boundaries After Training")
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        x = data_set.x
        y = data_set.y
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).sum().item() / len(y)
        print(f"\nFinal Accuracy: {accuracy*100:.2f}%")
    
    # Print class probabilities for the first few samples
    print("\nClass Probabilities for First 5 Samples:")
    softmax = nn.Softmax(dim=1)
    probs = softmax(outputs[:5])
    for i in range(5):
        print(f"Sample {i+1} - Class 0: {probs[i][0]:.4f}, "
              f"Class 1: {probs[i][1]:.4f}, "
              f"Class 2: {probs[i][2]:.4f}")
    
    # Print GPU memory usage if available
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

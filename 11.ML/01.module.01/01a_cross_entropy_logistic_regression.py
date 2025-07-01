"""
Logistic Regression Cross Entropy

Objective:
- How Cross-Entropy using random initialization influences the accuracy of the model.

Table of Contents:
1. Get Some Data
2. Create the Model and Total Loss Function
3. Train the Model via Batch Gradient Descent
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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

# Helper class for visualization (converted from notebook)
class plot_error_surfaces:
    def __init__(self, w_range, b_range, X, Y, n_samples=30, go=True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)
        Z = np.zeros((30, 30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                yhat = 1 / (1 + np.exp(-1*(w2*self.x+b2)))
                Z[count1, count2] = -1*np.mean(self.y*np.log(yhat+1e-16) + (1-self.y)*np.log(1-yhat+1e-16))
                count2 += 1
            count1 += 1
            
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        
        if go:
            plt.figure(figsize=(7.5, 5))
            ax = plt.axes(projection='3d')
            ax.plot_surface(self.w, self.b, self.Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_title('Loss Surface')
            ax.set_xlabel('w')
            ax.set_ylabel('b')
            plt.show()
            
            plt.figure()
            plt.contour(self.w, self.b, self.Z)
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
    
    def set_para_loss(self, model, loss):
        self.n = self.n + 1
        self.W.append(list(model.parameters())[0].item())
        self.B.append(list(model.parameters())[1].item())
        self.LOSS.append(loss)
    
    def final_plot(self):
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c='r', marker='x', s=200, alpha=1)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
    
    def plot_ps(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(self.x, self.y, 'ro', label="training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label="estimated line")
        plt.plot(self.x, 1 / (1 + np.exp(-1 * (self.W[-1] * self.x + self.B[-1]))), label='sigmoid')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-0.1, 2))
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.legend()
        
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.title('Loss Surface Contour Iteration ' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()

def PlotStuff(X, Y, model, epoch, leg=True):
    plt.plot(X.numpy(), model(X).detach().numpy(), label=('epoch ' + str(epoch)))
    plt.plot(X.numpy(), Y.numpy(), 'r')
    if leg:
        plt.legend()

# Create Data class
class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-1, 1, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0], 1)
        self.y[self.x[:, 0] > 0.2] = 1
        self.len = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

# Create dataset and data loader
data_set = Data()

def train_model(epochs=100, learning_rate=0.1):
    # Create model
    model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
    
    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    loss_list = []
    
    for epoch in range(epochs):
        # Move data to device
        x, y = data_set.x.to(device), data_set.y.to(device)
        
        # Forward pass
        y_hat = model(x)
        loss = criterion(y_hat, y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss
        loss_list.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    
    return model

# Train the model
print("\nStarting training...")
trained_model = train_model(epochs=100, learning_rate=0.1)

# Print final model parameters
print("\nFinal model parameters:")
for name, param in trained_model.named_parameters():
    print(f"{name}: {param.data}")

# Print GPU memory usage if available
if torch.cuda.is_available():
    print("\nGPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

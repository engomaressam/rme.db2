# <Deeper Neural Networks with nn.ModuleList()>
# Objective for this Notebook:
# 1. Create a Deeper Neural Network with nn.ModuleList()
# 2. Train and Validate the Model.
# Table of Contents:
# - Neural Network Module and Function for Training
# - Train and Validate the Model
# Estimated Time Needed: 25 min
#
# Preparation
# We'll need the following libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader
import sys
import io
import os
from contextlib import redirect_stdout
from docx import Document
from docx.shared import Inches

script_name = os.path.splitext(os.path.basename(__file__))[0]
word_output = f"{script_name}.docx"
doc = Document()
doc.add_heading(f'Output for {script_name}', 0)

f = io.StringIO()
with redirect_stdout(f):
    torch.manual_seed(1)

# Function used to plot:
def plot_decision_regions_3class(model, data_set):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    _, yhat = torch.max(model(XX), 1)
    yhat = yhat.numpy().reshape(xx.shape)
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:] == 0, 0], X[y[:] == 0, 1], 'ro', label = 'y=0')
    plt.plot(X[y[:] == 1, 0], X[y[:] == 1, 1], 'go', label = 'y=1')
    plt.plot(X[y[:] == 2, 0], X[y[:] == 2, 1], 'o', label = 'y=2')
    plt.title("decision region")
    plt.legend()

# Create Dataset Class
class Data(Dataset):
    # Constructor
    def __init__(self, K=3, N=500):
        D = 2
        X = np.zeros((N * K, D))
        y = np.zeros(N * K, dtype='uint8')
        for j in range(K):
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 1, N)
            t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
            X[ix] = np.c_[r * np.sin(t), r*np.cos(t)]
            y[ix] = j
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.x = torch.from_numpy(X).type(torch.FloatTensor)
        self.len = y.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
    def plot_stuff(self):
        plt.plot(self.x[self.y[:] == 0, 0].numpy(), self.x[self.y[:] == 0, 1].numpy(), 'o', label="y = 0")
        plt.plot(self.x[self.y[:] == 1, 0].numpy(), self.x[self.y[:] == 1, 1].numpy(), 'ro', label="y = 1")
        plt.plot(self.x[self.y[:] == 2, 0].numpy(), self.x[self.y[:] == 2, 1].numpy(), 'go', label="y = 2")
        plt.legend()

# Neural Network Module using ModuleList()
class Net(nn.Module):
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = F.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation

# Function for training the model
def train(data_set, model, criterion, train_loader, optimizer, epochs=100):
    LOSS = []
    ACC = []
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
        ACC.append(accuracy(model, data_set))
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(LOSS, color = color)
    ax1.set_xlabel('Iteration', color = color)
    ax1.set_ylabel('total loss', color = color)
    ax1.tick_params(axis = 'y', color = color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color = color)
    ax2.plot(ACC, color = color)
    ax2.tick_params(axis = 'y', color = color)
    fig.tight_layout()
    plt.savefig(f"{script_name}_loss_acc.png")
    plt.close()
    return LOSS

def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).numpy().mean()

# Train and Validate the Model
data_set = Data()
data_set.plot_stuff()
plt.savefig(f"{script_name}_data.png")
plt.close()
data_set.y = data_set.y.view(-1)

# Train the model with 1 hidden layer with 50 neurons
Layers = [2, 50, 3]
model = Net(Layers)
learning_rate = 0.10
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=data_set, batch_size=20)
criterion = nn.CrossEntropyLoss()
LOSS = train(data_set, model, criterion, train_loader, optimizer, epochs=100)
plot_decision_regions_3class(model, data_set)
plt.savefig(f"{script_name}_decision1.png")
plt.close()

# Train the model with 2 hidden layers with 20 neurons
Layers = [2, 10, 10, 3]
model = Net(Layers)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=data_set, batch_size=20)
criterion = nn.CrossEntropyLoss()
LOSS = train(data_set, model, criterion, train_loader, optimizer, epochs=1000)
plot_decision_regions_3class(model, data_set)
plt.savefig(f"{script_name}_decision2.png")
plt.close()

# Practice: Create a network with three hidden layers each with ten neurons.
# Layers = [2, 10, 10, 10, 3]
# model = Net(Layers)
# learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
# train_loader = DataLoader(dataset = data_set, batch_size = 20)
# criterion = nn.CrossEntropyLoss()
# LOSS = train(data_set, model, criterion, train_loader, optimizer, epochs = 1000)
# plot_decision_regions_3class(model, data_set)

# End of script 

output_text = f.getvalue()
doc.add_heading('Console Output', level=1)
doc.add_paragraph(output_text)

for img in [f"{script_name}_loss_acc.png", f"{script_name}_data.png", f"{script_name}_decision1.png", f"{script_name}_decision2.png"]:
    if os.path.exists(img):
        doc.add_picture(img, width=Inches(5.5))
        os.remove(img)

doc.save(word_output)
print(f"All output and plots saved to {word_output}") 
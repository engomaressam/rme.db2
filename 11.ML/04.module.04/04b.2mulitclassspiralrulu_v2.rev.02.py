import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ... existing code ...
script_name = os.path.splitext(os.path.basename(__file__))[0]
word_output = f"{script_name}.docx"
# ... existing code ...
# Train the model with 3 hidden layers, 15 neurons each (rev.02)
Layers = [2, 15, 15, 15, 3]
model = Net(Layers)
learning_rate = 0.02  # changed from 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=data_set, batch_size=20)
criterion = nn.CrossEntropyLoss()
LOSS = train(data_set, model, criterion, train_loader, optimizer, epochs=800)
plot_decision_regions_3class(model, data_set)
plt.savefig(f"{script_name}_decision2.png")
plt.close()
# ... existing code ... 
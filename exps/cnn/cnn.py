import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_shape=144000, n_kernels=32, kernel_sizes=[5, 5], hidden_size=32, dropout=0.5, n_classes=6):
        super(CNN1D, self).__init__()
        
        self.layers = nn.ModuleList()
        # Input feature is expected to have shape (batch_size, channels, length), hence input channels=1.
        in_channels = 1
        
        for size in kernel_sizes:
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels, n_kernels, kernel_size=size, padding='same'),
                nn.BatchNorm1d(n_kernels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.layers.append(conv_layer)
            # Next layer's in_channels should be the current layer's out_channels
            in_channels = n_kernels
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_kernels * input_shape, hidden_size)
        self.fc_bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        for layer in self.layers:
            x = layer(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc_bn(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

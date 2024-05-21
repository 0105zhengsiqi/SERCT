import torch.nn as nn

class TemporalCNNBiLSTM(nn.Module):
    def __init__(self, num_classes, input_size=1, hidden_size=60):
        super(TemporalCNNBiLSTM, self).__init__()
        
        # Temporal Convolutional Network
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_size,
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )

        self.dropout = nn.Dropout(p=0.5)
        
        # Fully Connected Layer
        self.fc = nn.Linear(in_features=2*hidden_size, out_features=num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        # Temporal Convolutional Network
        x = self.cnn(x)
        
        # Bidirectional LSTM
        x = x.transpose(1, 2)
        x, _ = self.bilstm(x)

        x = self.dropout(x)
        
        # Fully Connected Layer
        x = self.fc(x[:, -1, :])
        
        return x
    

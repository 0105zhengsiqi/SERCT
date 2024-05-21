import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, rnn_size=8000, hidden_size=32, dropout=0.5, n_classes=7):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=rnn_size, hidden_size=rnn_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])  # Select the last time step's output
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

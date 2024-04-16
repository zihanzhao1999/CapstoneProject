'''
This is our LSTM model
'''

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Second LSTM layer - Takes input from the first LSTM layer
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Linear layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)  # Intermediate linear layer
        self.fc2 = nn.Linear(hidden_size // 2, output_size)  # Final output layer

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        
        # Pass through the first LSTM layer
        out, _ = self.lstm1(x, (h0, c0))
        # Pass the output of the first LSTM layer to the second LSTM layer
        out, _ = self.lstm2(out, (h0, c0))
        
        # Passing the output of the last time step through linear layers
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out
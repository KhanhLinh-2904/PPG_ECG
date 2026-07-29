import torch
import torch.nn as nn

class FocusedNeuralNetwork(nn.Module):
    def __init__(self):
        super(FocusedNeuralNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()  #threshold  [0, 1]
        )

    def forward(self, x):
      
        threshold = self.network(x)  # [batch_size, 1]
        
        return threshold
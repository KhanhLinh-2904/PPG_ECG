import torch
import torch.nn as nn

class FocusedNeuralNetwork(nn.Module):
    def __init__(self):
        super(FocusedNeuralNetwork, self).__init__()

        self.rr_branch = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32), 
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.pratio_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32), 
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(32 + 32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  
            nn.Sigmoid()          
        )

    def forward(self, x):
        rr_features = x[:, :3]   
        p_ratio = x[:, 3:4]      

        out_rr = self.rr_branch(rr_features)       
        out_pratio = self.pratio_branch(p_ratio)     

        combined_features = torch.cat((out_rr, out_pratio), dim=1) 

        threshold = self.regressor(combined_features) 
        return threshold
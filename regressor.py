import torch
import torch.nn as nn

class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        # self.bnorm1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 32)
        # self.bnorm2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        
        x = self.relu(self.fc2(x))
        return self.fc3(x)
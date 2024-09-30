# model class

import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, input_dim=22, output_dim=8, dropout=True):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        if dropout:
            self.dropout = nn.Dropout(0.2)
        else:
            self.dropout = None

        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.dropout:
            x = self.dropout(x)
        x = F.softmax(self.fc4(x), dim=1)
        return x
    
    def predict(self, x):
        # Set model to evaluation mode
        # Predict the label of the input features x
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            _, predicted = torch.max(output, 1)
        return predicted
    
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    
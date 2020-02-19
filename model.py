import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self,N_depths,N_lambda):
        super(Network, self).__init__()

        #This is the network that takes 7 parameters sampled at N_depths and tries to learn
        #the forward problem, that is, to synthesize, for start
        #Stokes I at N_lambda wavelengths 
        
        self.C1 = nn.Conv1d(7,7,3)
        self.P1 = nn.Pool1d(2)
        self.C2 = nn.Conv1d(7,14,3)
        self.P2 = nn.Pool1d(2)
        self.F1 = nn.Flatten()

        self.C2 = nn.Linear(hidden_size, hidden_size)
        self.C3 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.C1(x)
        out = self.relu(out)
        out = self.C2(out)
        out = self.relu(out)
        out = self.C3(out)
            
        return out
    
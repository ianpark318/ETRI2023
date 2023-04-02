import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim



class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)
    
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim)
        )
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:,-1])
        return x

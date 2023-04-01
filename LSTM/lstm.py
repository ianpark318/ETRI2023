import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dataset
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

import torchinfo
import numpy as np




class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=3, bidirectional=True, action_classes=15, dropout=0.4,
                 device=0):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.device = device

        if bidirectional:
            self.D = 2
        else:
            self.D = 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)

        self.fc_act = nn.Linear(hidden_size * self.D, action_classes)

    def forward(self, x):

        h_0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(self.device)
        c_0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(self.device)

        output, (hidden_state, cell_state) = self.lstm(x, (h_0, c_0))

        if self.bidirectional:
            output = output.view(-1, self.hidden_size * 2)

        else:
            output = output.view(-1, self.hidden_size)

        out = self.fc_act(output)

        return out


class LSTM_Gang(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(LSTM_Gang, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            # dropout = 0.1,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        # 학습 초기화를 위한 함수

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim))

    # 예측을 위한 함수
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
sequence_length = 240
input_size = 2

# input_data = np.random.rand(batch_size, sequence_length, input_size)
# input_tensor = torch.tensor(input_data, dtype=torch.float32, device=device)


model = LSTM(input_size=input_size).to(device)

torchinfo.summary(model, (1, 240, 2), device=device)


# ==============TEST==============
#
# class LSTMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#         super(LSTMModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.layer_dim = layer_dim
#         self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
#         self.fc = nn.Linear(64 * hidden_dim, output_dim)
#
#     def forward(self, x):
#         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
#         c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
#         out = torch.flatten(out,start_dim= 1, end_dim= 2)
#         out = self.fc(out)
#         return out
#
# # model = LSTMModel(8, 18, 7, 9)
# # # RuntimeError: For unbatched 2-D input, hx and cx should also be 2-D but got (3-D, 3-D) tensors
# #
# # x = torch.randn(1, 64, 8)
# # out = model(x) # works
# #
# # torchinfo.summary(model, (1, 64, 8), device="cpu")


## 드론 사진 객체 검출
## RGD pose 추정 , 3D key point detection - 수어 인식 관련 프로젝트

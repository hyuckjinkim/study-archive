import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class BiLSTM(nn.Module):
    def __init__(self, bidirectional, batch_norm, input_size, hidden_size, num_layers, output_size, dropout, seq_len):
        super(BiLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.batch_norm = batch_norm
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if self.bidirectional else 1
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=self.bidirectional))
        for i in range(num_layers-1):
            self.layers.append(
                nn.LSTM(hidden_size*self.num_directions, hidden_size, batch_first=True, bidirectional=self.bidirectional)
            )
        
        self.fc = nn.Linear(hidden_size*self.num_directions, output_size)
        self.dropout = nn.Dropout(dropout)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(seq_len)     #LSTM Layer 뒤에 있을 때
            self.bn2 = nn.BatchNorm1d(output_size) #Fully Connected 뒤에 있을 때
        self.activation = nn.ELU()
        
        # # 초기 가중치 설정
        # self.init_weights()
        
    def init_weights(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_normal_(param.data)
                elif 'bias' in name:
                    init.constant_(param.data, 0)
                
        init.xavier_normal_(self.fc.weight)
        init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # 초기 hidden state와 cell state를 0으로 초기화
        h0 = torch.zeros(self.num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_directions, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out = x
        for i in range(self.num_layers):
            out, (hn, cn) = self.layers[i](out, (h0, c0))
            out = self.dropout(out)
            if self.batch_norm:
                out = self.bn1(out)
            out = self.activation(out)
        
        # Fully connected layer
        out = out[:, -1, :] # 마지막 타임스텝의 hidden state를 선택
        out = self.fc(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.activation(out)
        
        return out
    
# class BiGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, seq_len):
#         super(BiGRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.bidirectional = True
#         self.num_directions = 2 if self.bidirectional else 1
        
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=self.bidirectional))
#         for i in range(num_layers-1):
#             self.layers.append(nn.GRU(hidden_size*self.num_directions, hidden_size, batch_first=True, bidirectional=self.bidirectional))
        
#         self.fc = nn.Linear(hidden_size*self.num_directions, output_size)
#         self.dropout = nn.Dropout(dropout)
#         self.bn = nn.BatchNorm1d(seq_len)
#         self.activation = nn.ELU()

#     def forward(self, x):
#         # 초기 hidden state를 0으로 초기화
#         h0 = torch.zeros(self.num_directions, x.size(0), self.hidden_size).to(x.device)

#         # Forward propagate GRU
#         out = x
#         for i in range(self.num_layers):
#             out, h_n = self.layers[i](out, h0)
#             out = self.dropout(out)
#             out = self.bn(out)
        
#         # Fully connected layer
#         out = out[:, -1, :] # 마지막 타임스텝의 hidden state를 선택
#         out = self.fc(out)
#         out = self.activation(out)
        
#         return out
"""
Classification Head to be used downstream from wav2vec features
"""
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import LSTM

#### Base LSTM Classification HEAD


class LSTMClassificationHead(nn.Module):
    """LSTM Head for wav2vec classification task."""

    def __init__(self, config, lstm_hidden_dimension: int, num_layers: int = 1, dropout: float = 0, bidirectional: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.lstm_hidden_dimension = lstm_hidden_dimension
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm = LSTM(config.hidden_size, lstm_hidden_dimension, num_layers = self.num_layers, dropout = self.dropout, bidirectional = self.bidirectional)
        
        if self.bidirectional:
            self.dense1 = nn.Linear(2*self.lstm_hidden_dimension, 2*self.lstm_hidden_dimension)
            self.dense2 = nn.Linear(2*self.lstm_hidden_dimension, 1)
        else:
            self.dense1 = nn.Linear(self.lstm_hidden_dimension, self.lstm_hidden_dimension)
            self.dense2 = nn.Linear(self.lstm_hidden_dimension, 2)
       
        self.drop = nn.Dropout(p=0.5)

    def forward(self, features, **kwargs):
        # Detect legnth of wav2vec presentatation
        processed_features, _ = self.lstm(features)
        out_forward = processed_features[:, 0, :]
        if self.bidirectional:
            out_reverse = processed_features[:, -1, :]
            out = torch.cat((out_forward, out_reverse), 1)
        else: 
            out = out_forward

        ## Feed into fully connected layers: 
        out = self.dense1(out)
        # Add dropout
        out = self.drop(out)
        out = self.dense2(out)
        #out = torch.sigmoid(out)
        #print(out.shape)

        return out


### Configure Multilevel LSTM 



## (transfromer based?)
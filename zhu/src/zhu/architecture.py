import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EEGNet(nn.Module):
    def __init__(self, F1=16, eegnet_kernel_size=32, D=2, eeg_chans=22, eegnet_separable_kernel_size=16,
                 eegnet_pooling_1=8, eegnet_pooling_2=4, dropout=0.5):
        super(EEGNet, self).__init__()

        # self.F1 = F1
        # self.eegnet_kernel_size = eegnet_kernel_size
        # self.D = D
        F2 = F1*D
        self.dropout = nn.Dropout(dropout)

        self.block1 = nn.Conv2d(1, F1, (1, eegnet_kernel_size), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        # self.block2 = nn.Conv2d(F1, F1*D, (eeg_chans, 1), groups=F1, padding='valid', bias=False)
        self.block2 = nn.Conv2d(F1, F1*D, (eeg_chans, 1), padding='valid', bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, eegnet_pooling_1))
        # self.block3 = nn.Conv2d(F1*D, F2, (1, eegnet_separable_kernel_size), padding='same', groups=F1*D, bias=False)
        self.block3 = nn.Conv2d(F1*D, F2, (1, eegnet_separable_kernel_size), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F1*D)
        self.avg_pool2 = nn.AvgPool2d((1, eegnet_pooling_2))

    def forward(self, x):
        # x.shape = (B, 1, C, L)

        x = self.block1(x)
        # x.shape = (B, F1, C, L)
        # if debug_mode_flag: print('Shape of x after block1 of EEGNet: ', x.shape)

        x = self.bn1(x)
        x = self.block2(x)
        # x.shape = (B, F1*D, 1, L)
        # if debug_mode_flag: print('Shape of x after block2 of EEGNet: ', x.shape)

        x = self.bn2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout(x)
        # x.shape = (B, F1*D, 1, L/8)
        # if debug_mode_flag: print('Shape of x before block3 of EEGNet: ', x.shape)
        x = self.block3(x)
        # x.shape = (B, F1*D, 1, L/8)
        # if debug_mode_flag: print('Shape of x after block3 of EEGNet: ', x.shape)

        x = self.bn3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout(x)
        # x.shape = (B, F1*D, 1, L/64)
        # if debug_mode_flag: print('Shape of x by the end of EEGNet: ', x.shape)

        return x

class EEGTransformerNet(nn.Module):
    def __init__(self, nb_classes, sequence_length, eeg_chans=22,
                 F1=16, D=2, eegnet_kernel_size=32, dropout_eegnet=0.3, eegnet_pooling_1=5, eegnet_pooling_2=5, 
                 MSA_num_heads = 8, flag_positional_encoding=True, transformer_dim_feedforward=2048, num_transformer_layers=6,
                 device='cpu'):
        super(EEGTransformerNet, self).__init__()
        """
        F1 = the number of temporal filters
        F2 = number of spatial filters
        """

        F2 = F1 * D
        self.sequence_length_transformer = sequence_length//eegnet_pooling_1//eegnet_pooling_2

        self.eegnet = EEGNet(eeg_chans=eeg_chans, F1=F1, eegnet_kernel_size=eegnet_kernel_size, D=D, 
                             eegnet_pooling_1=eegnet_pooling_1, eegnet_pooling_2=eegnet_pooling_2, dropout=dropout_eegnet)
        self.linear = nn.Linear(self.sequence_length_transformer, nb_classes)
        
        self.flag_positional_encoding = flag_positional_encoding
        self.pos_encoder = PositionalEncoding(self.sequence_length_transformer, dropout=0.3)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.sequence_length_transformer,
            nhead=MSA_num_heads,
            dim_feedforward=transformer_dim_feedforward
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_transformer_layers
        )
        self.device = device

    def forward(self, x):
        # input x shape: (batch_size, num_channels, seq_len) = (batch_size, 22, 1000)
        x = torch.unsqueeze(x, 1)
        # x = x.permute(0, 2, 3, 1)  # similar to Keras Permute layer
        ## expected input shape for eegnet is (batch_size, 1, num_channels, seq_len)
        # print('Shape of x before EEGNet: ', x.shape)
        x = self.eegnet(x)
        # print('Shape of x after EEGNet: ', x.shape)
        x = torch.squeeze(x) ## output shape is (Batch size, F1*D, L//pool_1//pool2))

        ### Transformer Encoder Module
        x = x.permute(2, 0, 1) # output shape: (seq_len, batch_size, F1*D)
        seq_len_transformer, batch_size_transformer, channels_transformer = x.shape
        x = torch.cat((torch.zeros((seq_len_transformer, batch_size_transformer, 1), 
                                   requires_grad=True).to(self.device), x), 2)
        x = x.permute(2, 1, 0) # ouptut shape: (channels+1, batch_size, seq_len). seq_len is seen as the embedding size
        # if debug_mode_flag: print('Shape of x before Transformer: ', x.shape)
        if self.flag_positional_encoding:
            x = x * math.sqrt(self.sequence_length_transformer)
            x = self.pos_encoder(x) ## output matrix shape: (channels+1, batch_size, seq_len)
        # if debug_mode_flag: print('Positional Encoding Done!')
        # if debug_mode_flag: print('Shape of x after Transformer: ', x.shape)
        # x = self.transformer(x)
        x = self.transformer_encoder(x)  # shape: (channels+1, batch_size, seq_len)
        x = x[0,:,:].reshape(batch_size_transformer, -1) # shape: (batch_size, seq_len)

        ### Linear layer module
        # if debug_mode_flag: print('Shape of x before linear layer: ', x.shape)
        x = self.linear(x)
        return x
from argparse import ArgumentParser
import os
import torch
from torch import nn


class MyLSTM(nn.Module):
    def __init__(self,
                 num_inputs: int = 9,
                 num_outputs: int = 8,
                 num_layers: int = 1,
                 hidden_size: int = 512
                 ):
        super(MyLSTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=self.num_inputs,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers)

        self.h0 = torch.randn(self.num_layers, 1, self.hidden_size)
        self.c0 = torch.randn(self.num_layers, 1, self.hidden_size)

        self.fc = nn.Linear(self.hidden_size, num_outputs)

    def init_sequence(self, batch_size, device):
        """Initializing the state."""
        self.device = device
        self.batch_size = batch_size
        # reset state
        h = self.h0.clone().repeat(1, self.batch_size, 1).to(self.device)
        c = self.c0.clone().repeat(1, self.batch_size, 1).to(self.device)
        self.state = (h, c)

    def forward(self, x=None):
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs).to(self.device)

        x = x.unsqueeze(0)
        outp, self.state = self.lstm(x, self.state)

        o = torch.sigmoid(self.fc(outp))

        return o, self.state

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lstm_num_layers', type=int, default=1)
        parser.add_argument('--lstm_hidden_size', type=int, default=512)
        return parser

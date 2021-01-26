from argparse import ArgumentParser
import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim.rmsprop import RMSprop

from model_lstm import MyLSTM
from model_ntm import MyNTM


class MyModel(pl.LightningModule):
    def __init__(self,
                 **kwargs
                 ):
        super().__init__()
        # save the hyperparameters. Can be accessed by self.hparams.[variable name]
        self.save_hyperparameters()

        if self.hparams.model.lower() == 'lstm':
            network = MyLSTM(
                self.hparams.seq_width + 1,
                self.hparams.seq_width,
                self.hparams.lstm_num_layers,
                self.hparams.lstm_hidden_size,
            )
        elif self.hparams.model.lower() == 'ntm':
            network = MyNTM(
                self.hparams.seq_width + 1,
                self.hparams.seq_width,
                self.hparams.ntm_controller_size,
                self.hparams.ntm_controller_layers,
                self.hparams.ntm_num_heads,
                self.hparams.ntm_memory_n,
                self.hparams.ntm_memory_m
            )
            # raise NotImplementedError('NTMx is not yet implemented')
        else:
            raise NotImplementedError('{} is not yet implemented'.format(self.hparams.model.lower()))
        self.model = network

        self.criterion = torch.nn.BCELoss()

    def forward(self, data_in, data_out):
        # get shapes
        inp_seq_len = data_in.size(0)
        outp_seq_len, batch_size, _ = data_out.size()

        # New sequence
        self.model.init_sequence(batch_size, self.device)

        # Feed the sequence + delimiter
        for i in range(inp_seq_len):
            self.model(data_in[i])

        # Read the output (no input given)
        y_out = []
        for i in range(outp_seq_len):
            temp, _ = self.model()
            y_out.append(temp)

        # prepare for output
        y_out = torch.stack(y_out)
        if y_out.dim() == 4:
            y_out = y_out.squeeze(dim=1)

        return y_out

    def training_step(self, batch, batch_idx):
        loss, cost = self.on_step(batch, batch_idx)

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_cost', cost, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'cost': cost}

    def validation_step(self, batch, batch_idx):
        loss, cost = self.on_step(batch, batch_idx)

        self.log('valid_loss', loss, on_epoch=True)
        return {'loss': loss, 'cost': cost}

    def on_step(self, batch, batch_idx):
        # get the data and permute correctly
        data_in, data_out = batch
        data_in = data_in.permute(1, 0, 2)
        data_out = data_out.permute(1, 0, 2)

        # batch_size
        batch_size = data_out.size(1)

        # forward
        output = self(data_in, data_out)

        # get loss
        loss = self.criterion(output, data_out)

        # get cost
        y_out_binarized = output.clone().data
        y_out_binarized = (y_out_binarized > 0.5).float()
        # The cost is the number of error bits per sequence
        cost = torch.sum(torch.abs(y_out_binarized - data_out.data))

        return loss, cost.item() / batch_size

    def validation_epoch_end(self, outputs):
        # get data
        cost = ([x['cost'] for x in outputs])
        mean_cost = np.array(cost).mean()

        self.log('valid_cost', mean_cost, prog_bar=True)

    def configure_optimizers(self):
        optimizer = RMSprop(self.parameters(),
                            momentum=self.hparams.rmsprop_momentum,
                            alpha=self.hparams.rmsprop_alpha,
                            lr=self.hparams.rmsprop_lr)

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--rmsprop_lr', type=float, default=3e-5)
        parser.add_argument('--rmsprop_alpha', type=float, default=0.95)
        parser.add_argument('--rmsprop_momentum', type=float, default=0.99)
        parser.add_argument('--gradient_clip', type=float, default=10)

        return parser

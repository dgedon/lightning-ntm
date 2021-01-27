# Adapted from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
import random
import torch
import numpy as np
import os
from argparse import ArgumentParser
import warnings
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class CopyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, num_batches, batch_size, seq_min_len, seq_max_len, seq_width):

        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_min_len = seq_min_len
        self.seq_max_len = seq_max_len
        self.seq_width = seq_width

        self.seq_len = random.randint(self.seq_min_len, self.seq_max_len)

        self.counter = 0

    def __len__(self):
        # return length
        return self.num_batches * self.batch_size

    def __getitem__(self, idx):
        # change the seq_len for this batch
        if self.counter == 0:
            self.seq_len = random.randint(self.seq_min_len, self.seq_max_len)

        seq = np.random.binomial(1, 0.5, (self.seq_len, self.seq_width))
        seq = torch.from_numpy(seq)
        outp = seq.clone()

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(self.seq_len + 1, self.seq_width + 1)
        inp[:self.seq_len, :self.seq_width] = seq
        inp[self.seq_len, self.seq_width] = 1.0  # delimiter in our control channel

        # increase counter
        self.counter = 0 if self.counter >= self.batch_size else + 1

        return inp.float(), outp.float()


class CopyTaskDataModule(pl.LightningDataModule):
    def __init__(self,
                 seq_width: int = 8,
                 seq_min_len: int = 1,
                 seq_max_len: int = 20,
                 train_batch_size: int = 1,
                 eval_batch_size: int = 1,
                 train_batches_per_epoch: int = 200,
                 val_batches: int = 50,
                 dataloader_num_workers: int = 4,
                 **kwargs
                 ):
        super().__init__()
        # save hyperparameter
        self.seq_width = seq_width
        self.seq_min_len = seq_min_len
        self.seq_max_len = seq_max_len

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches = val_batches

        self.dataloader_num_workers = dataloader_num_workers

    def setup(self, stage='fit'):
        self.train_dataset = CopyDataset(
            self.train_batches_per_epoch,
            self.train_batch_size,
            self.seq_min_len,
            self.seq_max_len,
            self.seq_width
        )

        self.eval_dataset = CopyDataset(
            self.val_batches,
            self.eval_batch_size,
            self.seq_min_len,
            self.seq_max_len,
            self.seq_width
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.dataloader_num_workers,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--seq_width', type=int, default=8)
        parser.add_argument('--seq_min_len', type=int, default=1)
        parser.add_argument('--seq_max_len', type=int, default=20)
        parser.add_argument('--train_batch_size', type=int, default=1)
        parser.add_argument('--eval_batch_size', type=int, default=1)
        parser.add_argument('--train_batches_per_epoch', type=int, default=500)
        parser.add_argument('--val_batches', type=int, default=100)
        parser.add_argument('--dataloader_num_workers', type=int, default=4)

        return parser

# Adapted from https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
import random
import torch
import numpy as np
import os
from argparse import ArgumentParser
import warnings
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class RepeatCopyDataset(Dataset):
    def __init__(self,
                 num_batches,
                 batch_size,
                 seq_min_len,
                 seq_max_len,
                 repeat_min,
                 repeat_max,
                 seq_width):

        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_min_len = seq_min_len
        self.seq_max_len = seq_max_len
        self.repeat_min = repeat_min
        self.repeat_max = repeat_max
        self.seq_width = seq_width

        self.seq_len = random.randint(self.seq_min_len, self.seq_max_len)
        self.reps = random.randint(self.repeat_min, self.repeat_max)

        # normalisation constants
        self.reps_mean = (self.repeat_max + self.repeat_min) / 2
        reps_var = (((self.repeat_max - self.repeat_min + 1) ** 2) - 1) / 12
        self.reps_std = np.sqrt(reps_var)

        self.counter = 0

    def reps_normalize(self, reps):
        return (reps - self.reps_mean) / self.reps_std

    def __len__(self):
        # return length
        return self.num_batches * self.batch_size

    def __getitem__(self, idx):
        # change the seq_len and repetitions for this batch
        if self.counter == 0:
            self.seq_len = random.randint(self.seq_min_len, self.seq_max_len)
            self.reps = random.randint(self.repeat_min, self.repeat_max)

        seq = np.random.binomial(1, 0.5, (self.seq_len, self.seq_width))
        seq = torch.from_numpy(seq)

        # The input includes 2 additional channels for the delimiter (end of seq) and num-reps
        inp = torch.zeros(self.seq_len + 2, self.seq_width + 2)
        inp[:self.seq_len, :self.seq_width] = seq
        inp[self.seq_len, self.seq_width] = 1.0  # delimiter in our control channel
        inp[self.seq_len + 1, self.seq_width + 1] = self.reps_normalize(self.reps)  # number of reps

        outp = torch.zeros(self.seq_len * self.reps + 1, self.seq_width + 1)
        outp[:self.seq_len * self.reps, :self.seq_width] = seq.clone().repeat(self.reps, 1)
        outp[self.seq_len * self.reps, self.seq_width] = 1.0  # end marker

        # increase counter
        self.counter = 0 if self.counter >= self.batch_size else + 1

        return inp.float(), outp.float()


class RepeatTaskDataModule(pl.LightningDataModule):
    def __init__(self,
                 seq_width: int = 8,
                 seq_min_len: int = 1,
                 seq_max_len: int = 10,
                 repeat_min: int = 1,
                 repeat_max: int = 10,
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
        self.repeat_min = repeat_min
        self.repeat_max = repeat_max

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches = val_batches

        self.dataloader_num_workers = dataloader_num_workers

    def setup(self, stage='fit'):
        self.train_dataset = RepeatCopyDataset(
            self.train_batches_per_epoch,
            self.train_batch_size,
            self.seq_min_len,
            self.seq_max_len,
            self.repeat_min,
            self.repeat_max,
            self.seq_width
        )

        self.eval_dataset = RepeatCopyDataset(
            self.val_batches,
            self.eval_batch_size,
            self.seq_min_len,
            self.seq_max_len,
            self.repeat_min,
            self.repeat_max,
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
        # remaining parameters are same as in copy task
        parser.add_argument('--repeat_min', type=int, default=1)
        parser.add_argument('--repeat_max', type=int, default=10)

        return parser

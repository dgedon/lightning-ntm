from argparse import ArgumentParser
import os
import torch
import pytorch_lightning as pl

from data_copytask import CopyTaskDataModule
from model import MyModel
from model_lstm import MyLSTM
from model_ntm import MyNTM


def parse_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CopyTaskDataModule.add_model_specific_args(parser)
    parser = MyModel.add_model_specific_args(parser)

    parser = MyLSTM.add_model_specific_args(parser)
    parser = MyNTM.add_model_specific_args(parser)

    return parser.parse_args()


def main():
    # args
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', choices=['ntm', 'lstm'], default='ntm')
    args = parse_args(parser)

    # seed
    pl.seed_everything(args.seed)

    # data
    data_module = CopyTaskDataModule.from_argparse_args(args)

    # model
    model = MyModel(**vars(args))

    # training
    trainer = pl.Trainer.from_argparse_args(args, gradient_clip_val=args.gradient_clip) #, limit_train_batches=10)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()

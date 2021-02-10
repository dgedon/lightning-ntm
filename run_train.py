from argparse import ArgumentParser
import os
import torch
import pytorch_lightning as pl

from data_copytask import CopyTaskDataModule
from data_repeatcopytask import RepeatTaskDataModule
from model import MyModel
from model_lstm import MyLSTM
from model_ntm import MyNTM


def parse_args(parser=None):
    if parser is None:
        parser = ArgumentParser()
    # temp = parser.parse_args()

    parser = pl.Trainer.add_argparse_args(parser)
    parser = CopyTaskDataModule.add_model_specific_args(parser)
    parser = RepeatTaskDataModule.add_model_specific_args(parser)
    parser = MyModel.add_model_specific_args(parser)

    parser = MyLSTM.add_model_specific_args(parser)
    parser = MyNTM.add_model_specific_args(parser)

    return parser.parse_args()


def main():
    # args
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', choices=['ntm', 'lstm'], default='ntm')
    parser.add_argument('--task', choices=['copy', 'repeat-copy'], default='repeat-copy')
    args = parse_args(parser)

    # seed
    pl.seed_everything(args.seed)

    # data
    if args.task.lower() == 'copy':
        data_module = CopyTaskDataModule.from_argparse_args(args)
    elif args.task.lower() == 'repeat-copy':
        data_module = RepeatTaskDataModule.from_argparse_args(args)
    else:
        raise NotImplementedError

    # model
    model = MyModel(**vars(args))

    # training
    trainer = pl.Trainer.from_argparse_args(args, gradient_clip_val=args.gradient_clip) #, limit_train_batches=10)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()

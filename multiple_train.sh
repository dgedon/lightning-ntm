#!/bin/bash

# train lstm
MODEL=lstm
EPOCHS=4  0

for SEED in {1..10}
do
  printf '\n'
  printf "=%.0s" {1..40}
  printf "\nTraining %s with seed %s for %s epochs \n" $MODEL $SEED $EPOCHS
  printf "=%.0s" {1..40}
  printf '\n\n'

  python run_train.py --seed $SEED --model $MODEL \
  --train_batch_size 8 --eval_batch_size 8 --train_batches_per_epoch 500\
   --max_epochs $EPOCHS --gpus 1 --accelerator ddp
done

# train ntm
MODEL=ntm
EPOCHS=20

for SEED in {1..10}
do
  printf '\n'
  printf "=%.0s" {1..40}
  printf "\nTraining %s with seed %s for %s epochs \n" $MODEL $SEED $EPOCHS
  printf "=%.0s" {1..40}
  printf '\n\n'

  python run_train.py --seed $SEED --model $MODEL \
  --train_batch_size 8 --eval_batch_size 8 --train_batches_per_epoch 500\
   --max_epochs $EPOCHS --gpus 1 --accelerator ddp
done

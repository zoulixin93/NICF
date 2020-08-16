#!/usr/bin/env bash
~/pyenv3/bin/python ./launch.py -data_path ./data/data/ -environment env -T 40 -ST [5,10,20,40] -agent Train -FA FA -latent_factor 50 \
-learning_rate 0.001 -training_epoch 3000 -seed 145 -gpu_no 0 -inner_epoch 50 -rnn_layer 2 -gamma 0.8 -batch 50 -restore_model False

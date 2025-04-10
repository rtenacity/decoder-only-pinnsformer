#!/bin/bash

# Run train.py in the background and redirect output to train.out
nohup python train.py > train.out 2>&1 &

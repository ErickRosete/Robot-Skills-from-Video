#!/usr/bin/env python3
"""
Created on Tue Jun 30 20:02:01 2020
@author: Suresh, Erick, Jessica
"""
# ------------ Network constants  ------------ #
N_DOF_ROBOT = 9 # From kitchen_multitask_v0.py
VISUAL_FEATURES = 64
PLAN_FEATURES = 256
USE_LOGISTICS = True
N_MIXTURES = 10 # for gaussian distributions
N_LOGITS = 10 # for logistic distributions

# ------------ Training constants  ------------ #
N_EPOCH = 10
WINDOW_SIZE = 8
MIN_WINDOWS_SIZE = 8
MAX_WINDOW_SIZE = 15
VAL_BATCH_SIZE = 50
TRAIN_BATCH_SIZE = 8
FILES_TO_LOAD = 1 # Files to load during training simultaneously
EVAL_FREQ = 50 # Validate every eval_freq batches
LEARNING_RATE = 2e-4
BETA = 0.01

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
N_MIXTURES = 10

# ------------ Training constants  ------------ #
N_EPOCH = 20
WINDOW_SIZE = 16
VAL_BATCH_SIZE = 128
TRAIN_BATCH_SIZE = 8 
FILES_TO_LOAD = 5 # Files to load during training simultaneously
EVAL_FREQ = 20 # Validate every eval_freq batches
LEARNING_RATE = 2e-4
BETA = 0.01
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:02:01 2020

@author: suresh, erick, jessica
"""
# ------------ Network constants  ------------ #
N_DOF_ROBOT = 9 # refer kitchen_multitask_v0.py
VISUAL_FEATURES = 64
PLAN_FEATURES = 256

# ------------ Training constants  ------------ #
N_EPOCH = 5
WINDOW_SIZE = 16
VAL_BATCH_SIZE = 128
TRAIN_BATCH_SIZE = 8 
FILES_TO_LOAD = 1 #Files to load during training simultaneously
EVAL_FREQ = 20 #validate every eval_freq batches
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:05:31 2022

@author: pertsevm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import datetime, os
import argparse

#######################################################
########## max and min length of CDR loops ############
max_CDR123_ab_len = 77 #72 + 5'_' signs
#max_CDR3_ab_len = 47 #46 + '_' sign to join the loops
max_CDR_123b_len = 38 #36 + 2'_'signs
#max_CDR_3b_len = 23 #max len of CDR3b


#make a dict with data input ('beta') and columns to take&max_len
model_input_type = {}
keys = ['beta_chain', 'paired_chain']
values = [(['CDR1b','CDR2b','CDR3b'], max_CDR_123b_len), \
          (['CDR1a', 'CDR2a',  'CDR3a', 'CDR1b', 'CDR2b',  'CDR3b'], max_CDR123_ab_len)]
model_input_dict = dict(zip(keys, values))


########################################################################################################
########################################## Open TCRpMHC files #######################################

dat_dir = '../datasets/'
TCRpMHC_ESM2_data = data_dir + 'full_TCRaa_data_with_epitope_labels.csv'
esm_embed_dir = '../datasets/ESM_embeddings/'


##################################### OUTPUT DIRECTORY ###################################################

triplet_out_model= "../triplet_models/"
triplet_analysis_dir = "../triplet_analysis/"
triplet_analysis_networks = "../triplet_networks/"

###########################################################################################################

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

directory_TCRpMHC = '/Users/pertsevm/Desktop/VAE_triplet_yet_again_final/Scripts/AE_and_triplet/one_hot_on_ESM_dataset/ESM_embeddings/'
TCRpMHC_ESM2_data = directory_TCRpMHC + 'full_TCRaa_for_ESM2_embeddings_w_Vdomains.csv'
esm_embed_dir = '/Users/pertsevm/Desktop/VAE_triplet_yet_again_final/Scripts/AE_and_triplet/one_hot_on_ESM_dataset/ESM_embeddings/'

#small dataset
#TCRpMHC_new_data_90cdhit = directory_TCRpMHC + 'TCRpMHC_data_with_folds_90cdhit.csv'

#big dataset
#TCRpMHC_new_data_90cdhit_big_beta = directory_TCRpMHC + 'tc_hard_human_beta_90%cdhit_85_1000_TCRs_balanced.csv'
#TCRpMHC_new_data_95cdhit_big_beta = directory_TCRpMHC + 'tc_hard_human_beta_95%cdhit_85_1000_TCRs_balanced.csv'


##################################### OUTPUT DIRECTORY ###################################################

triplet_out_model= "/Users/pertsevm/Desktop/VAE_triplet_yet_again_final/Scripts/AE_and_triplet/one_hot_on_ESM_dataset/onehot_triplet_models/"
triplet_analysis_dir = "/Users/pertsevm/Desktop/VAE_triplet_yet_again_final/Scripts/AE_and_triplet/one_hot_on_ESM_dataset/onehot_triplet_analysis/"
triplet_analysis_networks = "/Users/pertsevm/Desktop/VAE_triplet_yet_again_final/Scripts/AE_and_triplet/one_hot_on_ESM_dataset/"

###########################################################################################################

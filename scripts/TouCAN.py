#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:39:47 2022

@author: pertsevm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow_addons as tfa
#import tensorflow_datasets as tfds


import csv
import os 
import argparse

#import other python files
import config_script
import model_architecture
import enc_decode
import plot_latent_space
import cluster_performance_functions

# parse the command line
if __name__ == "__main__":
    # initialize Argument Parser
    parser = argparse.ArgumentParser()    #help="Debug", nargs='?', type=int, const=1, default=7
    # TCR input parameters
    parser.add_argument("--input_type", help="beta_chain or paired_chain", type=str)
    parser.add_argument("--encoding_type", help="onehot or ESM", type=str) ## one-hot or ESM embeddings
    parser.add_argument("--esm_type", help="ESM1v or ESM2", type=str, default='ESM1v')
    parser.add_argument("--chain_type", help="TCR or ab_VC or ab_V", type=str, default='ab_V')

    # model parameters
    parser.add_argument("--embedding_space", type=int) ## embedding dimension for TCR
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type = int, default = 15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--triplet_loss", help="hard or semihard", type=str, default='semihard')
    
    #plot embeddings
    parser.add_argument("--plot_embedding", type = bool, default = False)


######################################## --> started here
# we then read the arguments from the comand line
args = parser.parse_args()

INPUT_TYPE = args.input_type  #beta_chain or paired_chain
ENCODING_TYPE = args.encoding_type
ESM_TYPE = args.esm_type
CHAIN_TYPE = args.chain_type

EMBEDDING_SPACE = args.embedding_space
EPOCHS = args.epochs
PATIENCE = args.patience
BATCH_SIZE = args.batch_size
LRATE = args.learning_rate
LOSS = args.triplet_loss # 'semihard' or 'hard'

PLOT_EMB = args.plot_embedding  
# we then read the arguments from the comand line
# DATA_SIZE = args.data_size

#triplet hyperparameters
LOSS = args.triplet_loss #'semihard' or 'hard'
PLOT_EMB = args.plot_embedding

########################################################################################
############################ Open TCRpMHC and encode them ##############################

# open the files and assign some constant variables
TCRpMHC = pd.read_csv(config_script.TCRpMHC_ESM2_data, header = 0)
epi_list = TCRpMHC['Epitope'].unique().tolist()
epi_color = plot_latent_space.palette_31[:len(epi_list)]
epi_color_dict = dict(zip(epi_list, epi_color))
TCRpMHC['color'] = TCRpMHC['Epitope'].map(epi_color_dict)

TCRpMHC['CDR123ab'] = TCRpMHC['CDR1a'] + TCRpMHC['CDR2a'] +  TCRpMHC['CDR3a'] + \
    TCRpMHC['CDR1b'] + TCRpMHC['CDR2b'] + TCRpMHC['CDR3b']
TCRpMHC['CDR123b'] = TCRpMHC['CDR1b'] + TCRpMHC['CDR2b'] + TCRpMHC['CDR3b']


if ENCODING_TYPE == 'onehot':      
    model_input_dict = config_script.model_input_dict
    model_input_key = INPUT_TYPE
    df_columns = model_input_dict[model_input_key][0] #get columns to encode
    max_seq_len = model_input_dict[model_input_key][1] #get max length for padding
    print ('df_columns: ', df_columns, 'max_seq_len: ', max_seq_len)

    # make a dist of one-hot encoded amino acids for data encoding:
    AA_list = list("ACDEFGHIKLMNPQRSTVWY_")
    AA_one_hot_dict = {char : l for char, l in zip(AA_list, np.eye(len(AA_list), k=0))}

if ENCODING_TYPE == 'ESM':
    if ESM_TYPE == 'ESM1v':
        if CHAIN_TYPE == 'TCR':
            embeddings = np.loadtxt(config_script.esm_embed_dir + 'albeta_VC_esm1v.txt', dtype = float)
        if CHAIN_TYPE == 'ab_VC':
            if INPUT_TYPE == 'beta_chain':
                embeddings = np.loadtxt(config_script.esm_embed_dir + 'beta_VC_esm1v.txt', dtype = float)
            if INPUT_TYPE == 'paired_chain':
                embeddings_b = np.loadtxt(config_script.esm_embed_dir + 'beta_VC_esm1v.txt', dtype = float)
                embeddings_a = np.loadtxt(config_script.esm_embed_dir + 'alpha_VC_esm1v.txt', dtype = float)
                embeddings = np.concatenate((embeddings_a, embeddings_b), axis=1) 
        if CHAIN_TYPE == 'ab_V':
            if INPUT_TYPE == 'beta_chain':
                embeddings = np.loadtxt(config_script.esm_embed_dir + 'beta_V_esm1v.txt', dtype = float)
            if INPUT_TYPE == 'paired_chain':
                embeddings_a = np.loadtxt(config_script.esm_embed_dir + 'alpha_V_esm1v.txt', dtype = float)
                embeddings_b = np.loadtxt(config_script.esm_embed_dir + 'beta_V_esm1v.txt', dtype = float)
                embeddings = np.concatenate((embeddings_a, embeddings_b), axis=1) 

    if ESM_TYPE == 'ESM2':
        if CHAIN_TYPE == 'TCR':
            embeddings = np.loadtxt(config_script.esm_embed_dir + 'albeta_VC_esm2.txt', dtype = float)
        if CHAIN_TYPE == 'ab_VC':
            if INPUT_TYPE == 'beta_chain':
                embeddings = np.loadtxt(config_script.esm_embed_dir + 'beta_VC_esm2.txt', dtype = float)
            if INPUT_TYPE == 'paired_chain':
                embeddings_a = np.loadtxt(config_script.esm_embed_dir + 'alpha_VC_esm2.txt', dtype = float)
                embeddings_b = np.loadtxt(config_script.esm_embed_dir + 'beta_VC_esm2.txt', dtype = float)
                embeddings = np.concatenate((embeddings_a, embeddings_b), axis=1) 
        if CHAIN_TYPE == 'ab_V':
            if INPUT_TYPE == 'beta_chain':
                embeddings = np.loadtxt(config_script.esm_embed_dir + 'beta_V_esm2.txt', dtype = float)
            if INPUT_TYPE == 'paired_chain':
                embeddings_a = np.loadtxt(config_script.esm_embed_dir + 'alpha_V_esm2.txt', dtype = float)
                embeddings_b = np.loadtxt(config_script.esm_embed_dir + 'beta_V_esm2.txt', dtype = float)
                embeddings = np.concatenate((embeddings_a, embeddings_b), axis=1) 

# K-fold Cross Validation model evaluation #when model parameters are already settled
#folds_list = TCRpMHC['part'].unique().tolist()      
tf.keras.utils.set_random_seed(1234) ## set ALL random seeds for the program
import random
all_results = pd.DataFrame()
test_val_folds = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0]]
for test_val in test_val_folds: #len(test_val) range(6)
    #test_val = random.sample(folds_list, 2)
    test_val_fold_str = str(test_val[0]) + str(test_val[1])
    #test_val_fold_str = str(fold[0]) + str(fold[1])
    val = TCRpMHC[TCRpMHC['part']==test_val[0]]
    test = TCRpMHC[TCRpMHC['part']==test_val[1]]
    train = TCRpMHC[~TCRpMHC['part'].isin(test_val)]
    
    #test = TCRpMHC[TCRpMHC['part']==5] #6

    #encode the data
    if ENCODING_TYPE == 'onehot':
        X_train = enc_decode.CDR_one_hot_encoding(train, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
        X_val = enc_decode.CDR_one_hot_encoding(val, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
        X_test = enc_decode.CDR_one_hot_encoding(test, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
        #X_all_data = enc_decode.CDR_one_hot_encoding(TCRpMHC, df_columns, AA_list, AA_one_hot_dict, max_seq_len)
        print ('Seqs encoded!')

    if ENCODING_TYPE == 'ESM':
        #1) get indexes
        train_idx = train.index.tolist()
        val_idx = val.index.tolist()
        test_idx = test.index.tolist()

        #2) get embeddings
        X_train = embeddings[train_idx]
        X_val = embeddings[val_idx]
        X_test = embeddings[test_idx]

        #3) change the dimension to train CNN
        dim_2 = int(len(X_train[0])/20) #get the dimension for reshaping
        X_train = np.reshape(X_train, (-1, dim_2, 20)) #it's either 64*20 or 128*20, depending on TCR input type
        X_val = np.reshape(X_val, (-1, dim_2, 20))
        X_test = np.reshape(X_test, (-1, dim_2, 20))

    y_train = train['label'].values
    y_val = val['label'].values
    y_test = test['label'].values

    #save the Epi label for plotting
    y_train_epi = train['Epitope'].tolist()
    y_val_epi = val['Epitope'].tolist()
    y_test_epi = test['Epitope'].tolist()

    print ('X_train: ', X_train.shape, \
           'X_val: ',  X_val.shape, \
           'X_test: ', X_test.shape)

    #turn array into tensors
    train_triplet = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_triplet = train_triplet.shuffle(1024).batch(BATCH_SIZE)
    val_triplet = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_triplet = val_triplet.batch(BATCH_SIZE)

######################################################################################
############################ Set up & train a model  #################################

    if LOSS == 'hard':
        loss = tfa.losses.TripletHardLoss()
    if LOSS == 'semihard':
        loss = tfa.losses.TripletSemiHardLoss()


    tf.keras.backend.clear_session()

    triplet = model_architecture.Triplet_CNN_naive(X_train.shape[1:], embedding_units = EMBEDDING_SPACE)
    triplet.compile(optimizer=tf.keras.optimizers.Adam(LRATE), loss=loss) #tfa.losses.TripletSemiHardLoss()
    triplet.summary()
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience = PATIENCE, \
                       verbose = 1, restore_best_weights = True)
    
    print ('Starting training...')
    history = triplet.fit(
        train_triplet,
        epochs=EPOCHS, 
        validation_data = val_triplet, 
        callbacks = [early_stop])

    print ('Model trained!')

    # Extract the best loss value (from early stopping) 
    loss_hist = history.history['val_loss']
    best_val_loss = np.min(loss_hist)
    best_epochs = np.argmin(loss_hist) + 1

    if ENCODING_TYPE == 'onehot':
        TRIPLET_NAME =f'triplet_{ENCODING_TYPE}_{INPUT_TYPE}_{EMBEDDING_SPACE}d_{LRATE}lr_{BATCH_SIZE}btch_{EPOCHS}ep_{PATIENCE}ptence_fold{test_val_fold_str}'
    if ENCODING_TYPE == 'ESM':
        TRIPLET_NAME =f'triplet_{INPUT_TYPE}_{ESM_TYPE}_{CHAIN_TYPE}_{EMBEDDING_SPACE}d_{LRATE}lr_{BATCH_SIZE}btch_{EPOCHS}ep_{PATIENCE}ptence_fold{test_val_fold_str}'

    modelpath = config_script.triplet_out_model + TRIPLET_NAME
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    # save the model
    triplet.save(config_script.triplet_out_model + TRIPLET_NAME)

    
    #print ('\n')
    print ('Model name: ', TRIPLET_NAME)
    print ('best loss: ', best_val_loss, 'best_epochs: ', best_epochs)
    print ('Triplet model saved!')
    
    ############################## extract clustering performance ###############################
    ############################## ################################ #############################
    if INPUT_TYPE == 'beta_chain':
        seq_column = 'CDR123b'
    if INPUT_TYPE == 'paired_chain':
        seq_column = 'CDR123ab'
    
    results_val = triplet(X_val)
    val_emb_numpy = results_val.numpy()
    results_test = triplet(X_test)
    test_emb_numpy = results_test.numpy()
    
    epsilon_threshold = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.85, 0.95]
    for epsilon in epsilon_threshold:
        result_dbscan_val = cluster_performance_functions.get_performance_DBSCAN(val, TRIPLET_NAME, \
                                                        val_emb_numpy, seq_column, epsilon)
        result_dbscan_val['split'] = 'val'
        result_dbscan_val['fold'] = test_val[0]
    
    
        result_dbscan_test = cluster_performance_functions.get_performance_DBSCAN(test, TRIPLET_NAME, \
                                                            test_emb_numpy, seq_column, epsilon)
        result_dbscan_test['split'] = 'test'
        result_dbscan_test['fold'] = test_val[1]

        all_results = pd.concat([all_results, result_dbscan_val, result_dbscan_test], ignore_index = True)


all_results.to_csv(config_script.triplet_analysis_dir + TRIPLET_NAME + '_WITH_FOLDS_clstr_metrics.csv')

"""
########################################################################################
############################## Plot model embeddings ###################################
triplet = keras.models.load_model(config_script.triplet_out_model + TRIPLET_NAME)  
if PLOT_EMB == True:
    
    results_val = triplet(X_val)
    results_test = triplet(X_test)
    results_train = triplet(X_train)

    umap_pca_test = plot_latent_space.umap_pca_df(results_test, y_test_epi, random_seed = 42)
    plot_latent_space.plot_umap_pca_latent(umap_pca_test, 'Epitope', color_dict = epi_color_dict, \
                         fig_title = f'Epitope distribution by {TRIPLET_MODE} model with {INPUT_TYPE} triplet {EMBEDDING_SPACE}embedding ', \
                         legend_title = 'Epitope', node_size = 70, \
                         save = True, figname = f'Epitope_{TRIPLET_NAME}.jpg')
"""








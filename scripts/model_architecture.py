#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:32:20 2022

@author: pertsevm
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


def Triplet_CNN_naive(input_shape, embedding_units):
    
    ## triplet model architecture
    triplet_loss_model = tf.keras.Sequential(
        [tf.keras.Input(shape=input_shape),
         tf.keras.layers.Conv1D(filters=128, kernel_size=2, padding='valid', activation='relu'),
         tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='valid', activation='relu'),
         tf.keras.layers.Conv1D(filters=16, kernel_size=2, padding='same', activation='relu'),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(embedding_units, activation=None), # No activation on final dense layer
         tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))] # L2 normalize embeddings
    )
    
    return triplet_loss_model




def Triplet_FFNN_naive(input_shape, embedding_units):

    ## triplet model architecture
    triplet_loss_model = tf.keras.Sequential(
        [tf.keras.Input(shape=input_shape),
         tf.keras.layers.Dense(1024, activation='relu'),
         tf.keras.layers.Dense(512, activation='relu'),
         tf.keras.layers.Dense(256, activation='relu'),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(embedding_units, activation=None), # No activation on final dense layer
         tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))] # L2 normalize embeddings
    )

    return triplet_loss_model
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:18:52 2022

@author: pertsevm
"""
import numpy as np
from sklearn.metrics import accuracy_score

AA_list = list("ACDEFGHIKLMNPQRSTVWY_")
AA_one_hot_dict = {char : l for char, l in zip(AA_list, np.eye(len(AA_list), k=0))}

# a function to encode the data as one-hot vectors
def CDR_one_hot_encoding(df, column_list, aa_list, aa_one_hot_dict, max_seq_len):
    
    # 0) shuffle the datasets just in case
    #from sklearn.utils import shuffle
    #df = shuffle(df)
    
    # 1) concatenate the loops together or take seqs from a single column
    if len(column_list) > 1:
        df['concat_loops'] = df[column_list].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        #print ('with dupl: ', len(df))
        #df = df[['concat_loops']].drop_duplicates()
        #print ('withOUT dupl: ', len(df))
        seqs_list = df['concat_loops'].tolist()
    else:
        seqs_list = df[column_list[0]].tolist()
    
    # 2) pad the shorter sequences with '_'
    padded_seqs = [seq + (max_seq_len - len(seq))*'_' for i, seq in enumerate(seqs_list)]
    
    # 3) encode sequences:
    sequences=[]
    for seq in padded_seqs: 
        encoded_seq=np.zeros((len(seq), len(aa_list)))
        count=0
        for aa in seq:
            if aa in aa_list:
                encoded_seq[count]=aa_one_hot_dict[aa]
                count+=1
            else:
                print ("Unknown amino acid in peptides: "+ aa +", encoding aborted!\n")
                #sys.stderr.write("Unknown amino acid in peptides: "+ aa +", encoding aborted!\n")
                #sys.exit(2)
        sequences.append(encoded_seq)
        
    enc_aa_seq = np.asarray(sequences)
    
    return enc_aa_seq


def recnstr_acc_all(data, predictions, flat_enc):
    #accuracy on all (with _)
    if flat_enc == True:
        predictions = [np.reshape(reconstr_seq, (-1, len(AA_list))) for reconstr_seq in predictions]
        data = [np.reshape(seq, (-1, len(AA_list))) for seq in data]

    accuracy_all = np.mean([accuracy_score(y_true = np.argmax(data[i], 1), 
                                   y_pred = np.argmax(predictions[i], 1)) for i in range(len(data))])
    return accuracy_all

def recnstr_acc_loops(data, predictions, flat_enc):
    # accuracy on loops, only on amino acids (no dash line)
    pred_loops = []
    true_loops = []
    count_unequal_len = 0
    
    if flat_enc == True:
        predictions = [np.reshape(reconstr_seq, (-1, len(AA_list))) for reconstr_seq in predictions]
        data = [np.reshape(seq, (-1, len(AA_list))) for seq in data]

    for i in range(len(data)):
        #extract amino acid indexes
        aa_idx_pred = np.argmax(predictions[i], 1)
        aa_idx_true = np.argmax(data[i], 1)
        
        #get indexes of AA !=20 (20 is a dash line)
        idx_pred = np.where(aa_idx_pred!=20)[0]
        idx_true = np.where(aa_idx_true!=20)[0]
        #get list of arrays of loops (which were split by the dash line)
        aout_pred = np.split(aa_idx_pred[idx_pred],np.where(np.diff(idx_pred)!=1)[0]+1) #split into loops (based on a dash line)
        aout_true = np.split(aa_idx_true[idx_true],np.where(np.diff(idx_true)!=1)[0]+1)
        
        #compare the length of the loops and only append loops of equal length (otherwise the accuracy wouldn't work)
        pred_len = [len(l) for l in aout_pred]
        true_len = [len(l) for l in aout_true]
        if pred_len == true_len:
            pred_loops.append(aout_pred)
            true_loops.append(aout_true)
        else: 
            count_unequal_len += 1
    
    print ('% of loops with unequal length: ', count_unequal_len/len(data))
    accuracy_values = []
    for k in range(len(aout_pred)): #iterate through every loop
        accuracy_ = np.mean([accuracy_score(y_true = true_loops[l][k], 
                                            y_pred = pred_loops[l][k]) for l in range(len(data)-count_unequal_len)])
        accuracy_values.append(accuracy_)
        
    return accuracy_values


def decode_cdr3(output_cdr3, aa_list, flat_enc):
    #decode cdr3
    decoded_cdr3_seqs = []
    for seq_array in output_cdr3:
        if flat_enc == True:
            #reshape linear 1d cdr3 array into 2d array of size [len_of_seq; len_of_encoding]
            seq_array = np.reshape(seq_array, (-1, len(AA_list))) #-1 means that the other dimension is inferred automatically
        #1)np.argmax(seq_array_2d, axis=1).tolist() - take the index of the max value in every row and make a list of indexes
        #2)find amino acid
        decoded_cdr3_seqs.append(''.join(AA_list[i] for i in np.argmax(seq_array, axis=1).tolist() )  )
    return decoded_cdr3_seqs
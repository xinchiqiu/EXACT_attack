from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import itertools
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from utils import *

from DeepCTR.deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat, get_feature_names)
#from DeepCTR.deepctr_torch.models.deepfm import *
from deepfm import *
from deepfm_early import *
import sys
import argparse


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
        print("DEVICE: USING CUDA")
    else:
        device = "cpu"
        print("DEVICE: USING CPU")
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_dp', type=int, required=False, default=0)
    parser.add_argument('--norm_clip', type=float, required=False, default=0.035)
    parser.add_argument('--noise', type=float, required=False, default=0.01)
    parser.add_argument('--use_label_dp', type=int, required=False, default=0)
    parser.add_argument('--label_dp_prob', type=float, required=False, default=0.1)
    parser.add_argument('--fl', type=int, required=False, default=0)
    args = parser.parse_args()
    use_dp = bool(args.use_dp)
    norm_clip = args.norm_clip
    noise = args.noise
    use_label_dp = bool(args.use_label_dp)
    label_dp_prob = args.label_dp_prob
    fl = bool(args.fl)
    print('loading adult marketing dataset')
    #client_features = ['gender','race','marital','relationship','occupation','workclass']
    #sparse_features = ['country','education']
    client_features = ['gender', 'race', 'relationship', 'marital']
    sparse_features = ['workclass','education','occupation','country'] 

    #client_features = ['education','occupation']
    #sparse_features = ['country','gender','race','marital', 'relationship','workclass']
    dense_features = ['age','gain', 'loss', 'hours_per_week']
    linear_feature_columns, dnn_feature_columns, client_feature_columns = get_adult_columns(client_features, sparse_features, dense_features)
    server_train,  client_train, y_train, server_test, client_test, y_test = load_adult(client_features, sparse_features, dense_features)

    #split learning
    print('init model...')
    server_model = DeepFM_server_early(linear_feature_columns, dnn_feature_columns, dnn_hidden_units = (256,128), task='binary', l2_reg_embedding=1e-4, device = device)
    client_model = DeepFM_client_early(client_feature_columns, client_feature_columns, 36, dnn_hidden_units = (256,128), task='binary', l2_reg_embedding=1e-4, device = device)
    server_model.compile('adagrad', 'binary_crossentropy',metrics=['binary_crossentropy','auc'])
    client_model.compile('adagrad', 'binary_crossentropy',metrics=['binary_crossentropy','auc'])
    print(f'start training...with fl={fl}')
    #print(server_model)
    #print(client_model)
    fit_split_deepfm(server_model, 
                    client_model, 
                    server_train,  
                    client_train, 
                    y_train, 
                    server_test, 
                    client_test,
                    y_test, 
                    device,
                    save_folder='adult', 
                    batch_size=1024, 
                    epochs=2, 
                    verbose =2,
                    use_dp=use_dp,
                    norm_clip=norm_clip,
                    noise=noise,
                    use_label_dp=use_label_dp,
                    label_dp_prob=label_dp_prob,
                    fl=fl)


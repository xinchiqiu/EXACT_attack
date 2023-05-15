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
    parser.add_argument('--norm_clip', type=float, required=False, default=0.005)
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
    print('loading dataset')
    linear_feature_columns, dnn_feature_columns, client_feature_columns = get_columns()
    server_train,  client_train, y_train, server_test, client_test, y_test = load_taobao_df()
    #model = DIN(linear_feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    #model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', l2_reg_embedding=1e-4, device = device)
    #model.compile('adam', 'binary_crossentropy',metrics=['binary_crossentropy','auc'])
    #history = model.fit(server_train, y_train, batch_size=1024, epochs=10, verbose=2, validation_split=0.2)
    #fit_deepfm(model, device, server_train, client_train, y_train, server_test, client_test, y_test, batch_size = 256, epochs = 5, verbose = 2)

    #split learning
    print('init model...')
    server_model = DeepFM_server_early(linear_feature_columns, dnn_feature_columns, task='binary', l2_reg_embedding=1e-4, device = device)
    client_model = DeepFM_client_early(client_feature_columns, client_feature_columns, 41, task='binary', l2_reg_embedding=1e-4, device = device)
    server_model.compile('adagrad', 'binary_crossentropy',metrics=['binary_crossentropy','auc'])
    client_model.compile('adagrad', 'binary_crossentropy',metrics=['binary_crossentropy','auc'])
    print(f'start training...with fl={fl}')
    fit_split_deepfm(server_model, 
                    client_model, 
                    server_train,  
                    client_train, 
                    y_train, 
                    server_test, 
                    client_test,
                    y_test, 
                    device, 
                    save_folder='taobao',
                    batch_size=1024, 
                    epochs=2, 
                    verbose =2,
                    use_dp=use_dp,
                    norm_clip=norm_clip,
                    noise=noise,
                    use_label_dp=use_label_dp,
                    label_dp_prob=label_dp_prob,
                    fl=fl)


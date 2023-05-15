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
from DeepCTR.deepctr_torch.inputs import (DenseFeat, SparseFeat, get_feature_names)
#from DeepCTR.deepctr_torch.models.deepfm import *
from deepfm import *
import sys
import pickle
import argparse
from attack_taobao import Reconstruct_deepfm

def attack_bank(testloader, 
                cluster_combination_list,
                server_model, 
                client_model, 
                loss_fn, 
                use_label_dp = False, 
                label_dp_prob = 0.1, 
                use_dp = False,
                norm_clip = 1, 
                noise = 0.0005,
                if_save = False,
                if_load = False,
                load_from = None, 
                fl = False,
                num_features=5,
    ):
    stats_output= {}
    num_samples = 0
    # initiaze the reconstruct machine
    recon_machine = Reconstruct_deepfm(
        server_model = server_model,
        client_model = client_model
    )
    if if_load:
        if use_dp and use_label_dp:
            name_o = 'attack/banking/attack_original_dp' +str(noise)+'_'+ str(load_from) + '_labeldp'+str(label_dp_prob)+'.pkl'
            name_r = 'attack/banking/attack_recon_dp'  +str(noise)+'_'+ str(load_from) + '_labeldp'+str(label_dp_prob)+'.pkl'
        elif use_dp and not use_label_dp:
            name_o = 'attack/banking/attack_original_dp' +str(noise)+'_'+ str(load_from) + '.pkl'
            name_r = 'attack/banking/attack_recon_dp' + str(noise)+'_'+ str(load_from) + '.pkl'
        elif use_label_dp and not use_dp:
            name_o = 'attack/banking/attack_original_labeldp' +str(label_dp_prob)+'_'+ str(load_from) + '.pkl'
            name_r = 'attack/banking/attack_recon_labeldp' + str(label_dp_prob)+'_'+ str(load_from) + '.pkl'
        elif fl:
            name_o = 'attack/banking/attack_original_fl' + str(load_from)+ '.pkl'
            name_r = 'attack/banking/attack_recon_fl' + str(load_from)+ '.pkl'
        else:
            name_o = 'attack/banking/attack_original_' + str(load_from)+ '.pkl'
            name_r = 'attack/banking/attack_recon_' + str(load_from)+ '.pkl'
        print('loading attack pickle..')
        print(name_o)
        print(name_r)
        with open(name_o, 'rb') as f:
            recon_machine.original = pickle.load(f)
        with open(name_r, 'rb') as f:
            recon_machine.recon = pickle.load(f)
        print('length of loading:', len(recon_machine.original))
  
    server_model.eval()
    client_model.eval()

    for server_test, client_test, y_test in testloader:
        num_samples += 1
        
        if num_samples > load_from:
            server_test = server_test.to(device).float()
            client_test = client_test.to(device).float()
            y_test = y_test.to(device).float()
            cate_original = client_test
            y_test_original = y_test.clone()

            # need to flip the label here to get the dx,
            # but in the attack feed in the original, since it will go through both label to attack
            if use_label_dp:
                y_test, _ = label_dp(y_test, label_dp_prob)
                y_test = y_test.to(device)
            
            server_output_original = server_model(server_test)
            client_output_original = client_model(server_output_original,client_test)
            
            # to get the dx_original use the flipped label
            _, dx_original = client_model.backward(server_output_original, client_output_original, y_test, loss_fn, update_model_grad=False)
            dx_original = dx_original[0]

            if use_dp:
                dx_original, _ = dp(dx = dx_original, clip= norm_clip, noise=noise, device=device)
            
            
            # Attack
            # while doing the attack, label is flipped, but compared to get the accuracy
            _ = recon_machine.reconstruct(
                cluster_combination_list,
                server_output_original,
                y_test_original, #want to compare to the true label
                dx_original,
                cate_original,
                loss_fn,
            )
            
            if num_samples % 500 == 0:
                print('num_samples', num_samples)
                if if_save:
                    print(f'saving the attack results')
                    if use_dp and use_label_dp:
                        name_o = 'attack/banking/attack_original_dp' +str(noise)+'_'+ str(num_samples) + '_labeldp'+str(label_dp_prob)+'.pkl'
                        name_r = 'attack/banking/attack_recon_dp'  +str(noise)+'_'+ str(num_samples) + '_labeldp'+str(label_dp_prob)+'.pkl'
                    elif use_dp and not use_label_dp:
                        name_o = 'attack/banking/attack_original_dp' +str(noise)+'_'+ str(num_samples) + '.pkl'
                        name_r = 'attack/banking/attack_recon_dp'  +str(noise)+'_'+ str(num_samples) + '.pkl'
                    elif use_label_dp and not use_dp:
                        name_o = 'attack/banking/attack_original_labeldp' +str(label_dp_prob)+'_'+ str(num_samples) + '.pkl'
                        name_r = 'attack/banking/attack_recon_labeldp'  +str(label_dp_prob)+'_'+ str(num_samples) + '.pkl'
                    elif fl:
                        name_o = 'attack/banking/attack_original_fl'+str(num_samples)+'.pkl'
                        name_r = 'attack/banking/attack_recon_fl'+str(num_samples)+'.pkl'
                    else:
                        name_o = 'attack/banking/attack_original_'+str(num_samples)+'.pkl'
                        name_r = 'attack/banking/attack_recon_'+str(num_samples)+'.pkl'
                    with open(name_o, 'wb') as f:
                        pickle.dump(recon_machine.original, f)
                    with open(name_r, 'wb') as f:
                        pickle.dump(recon_machine.recon, f)
            
                for i in range(num_features+1):
                    true = np.array(recon_machine.original)[:,i]
                    pred = np.array(recon_machine.recon)[:,i]
                    if i == 3 or i ==4 or i==6:
                        acc_score = metrics.accuracy_score(true,pred)
                        f1_score = metrics.f1_score(true,pred)
                        recall_score = metrics.recall_score(true, pred)
                        precision = metrics.precision_score(true, pred)
                    else:
                        acc_score = metrics.accuracy_score(true,pred)
                        f1_score = metrics.f1_score(true, pred, average='macro')
                        recall_score = metrics.recall_score(true, pred,  average='macro')
                        precision = metrics.precision_score(true, pred, average='macro')
                    print(f'recon on feature ({i}) acc = {acc_score}, f1_score = {f1_score}, recall= {recall_score}, precision= {precision}')

    return stats_output



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saving', type=int, required=False, default=1)
    parser.add_argument('--use_label_dp', type=bool, required=False, default=False)
    parser.add_argument('--use_dp', type=int, required=False, default=0)
    parser.add_argument('--label_dp_prob', type=float, required=False, default=0.1)
    parser.add_argument('--norm_clip', type=float, required=False, default=0.025)
    parser.add_argument('--noise', type=float, required=False, default=0.01)
    parser.add_argument('--fl', type=int, required=False, default=0)
    parser.add_argument('--loading', type=int, required=False, default=0)
    parser.add_argument('--load_from', type=int, required=False, default=0)
    args = parser.parse_args()
    use_label_dp = args.use_label_dp
    label_dp_prob = args.label_dp_prob
    use_dp = bool(args.use_dp)
    saving = bool(args.saving)
    norm_clip = args.norm_clip
    noise = args.noise
    loading = bool(args.loading)
    load_from = args.load_from
    fl = bool(args.fl)

    print(f'Starting with...DP = {use_dp} with norm clip {norm_clip} and noise {noise}')
    print(f'label dp = {use_label_dp} with probability = {label_dp_prob}')

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # prepare the dataset
    print('loading dataset')
    linear_feature_columns, dnn_feature_columns, client_feature_columns = get_banking_columns()
    server_train,  client_train, y_train, server_test, client_test, y_test = load_banking()

    # print(f'number of test set is = {len(y_test)}') #4522
    # load the model
    print('loading model')
    if use_dp and use_label_dp:
        server_f = 'model/banking/server_dp_1'+str(norm_clip)+str(noise)+'_labeldp'+str(label_dp_prob)+'.pt'
        client_f = 'model/banking/client_dp_1'+str(norm_clip)+str(noise)+'_labeldp'+str(label_dp_prob)+'.pt'
    elif use_dp and not use_label_dp:    
        server_f = 'model/banking/server_dp_1'+str(norm_clip)+str(noise)+'.pt'
        client_f = 'model/banking/client_dp_1'+str(norm_clip)+str(noise)+'.pt'
    elif fl:
        server_f = 'model/banking/server_model_1_fl.pt'
        client_f = 'model/banking/client_model_1_fl.pt'
    elif use_label_dp and not use_dp:
        server_f = 'model/banking/server_model_labeldp_1'+str(label_dp_prob)+'.pt'
        client_f = 'model/banking/client_model_labeldp_1'+str(label_dp_prob)+'.pt'
    else:
        server_f = 'model/banking/server_model_1.pt'
        client_f = 'model/banking/client_model_1.pt'
    
    print(server_f)
    print(client_f)
    server_model = torch.load(server_f)
    client_model = torch.load(client_f)
    
    # load testloader with batch size = 1
    print('load testloader')
    test_loader, _ = data_to_dataloader(server_model, server_test, client_test, y_test, 1, True)

    feature_offsite = ['marital', 'job', 'education', 'housing','loan','contact','label']
    num_features = len(feature_offsite) -1
    offsite_category = [3,12,4,2,2,3,2]
    cluster = []
    for i in range(len(offsite_category)):
        temp = [j for j in range(offsite_category[i])]
        cluster.append(temp)
    cluster_combination_list = list(itertools.product(*cluster))
    cluster_combination_list = torch.tensor(cluster_combination_list).to(device)
    print(f'attack on {num_features} features = {feature_offsite}')
    print(len(cluster_combination_list))

    # run the attack
    print('start attack...')
    loss_fn = client_model.loss_func
    output = attack_bank(
        test_loader, 
        cluster_combination_list, 
        server_model,
        client_model, 
        loss_fn, 
        use_label_dp=use_label_dp, 
        label_dp_prob = label_dp_prob, 
        use_dp = use_dp,
        norm_clip=norm_clip,
        noise=noise,
        if_save = saving,
        if_load = loading,
        load_from=load_from,
        fl=fl,
        num_features=num_features)


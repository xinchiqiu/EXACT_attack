from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.model_selection import train_test_split	
import itertools
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import argparse
from utils import *
import pickle
import argparse

class Reconstruct_taobao:
    def __init__(self, server_model, client_model):
        self.server_model = server_model
        self.client_model = client_model
        self.original = []
        self.recon = []

    def reconstruct(
        self,
        cluster_combination_list,
        server_output,
        label,
        dx_original,
        cate_original,
        offsite_feature_shape=5,
    ):
        self.client_model.eval()
        #cluster_combination_list = self.get_cluster_combination(cluster_centers)
        dx_diff_list = []

        for cate_try in cluster_combination_list:
            # forward pass the client side model
            # use the other data, but the reconstructed cluster center to be offsite feature to see if dx match.
            cate_try = torch.tensor(cate_try).type(torch.float32)
            cate_try = cate_try.unsqueeze(0)
            
            client_output_recon = client_model(server_output,cate_try)

            # backward pass the client side model
            # get the gradient and compare with dx and store
            
            _, dx_recon = client_model.backward(
                            server_output, 
                            client_output_recon, 
                            label,
                            loss_fn,
                            update_model_grad=False)

            dx_recon = dx_recon[0]
            # calculate the distance between dx and dx_original
            # to get the smallest distance as the reconstructed center
            dx_diff = self.get_dist_between_dx(dx_recon, dx_original)
            dx_diff_list.append(dx_diff.clone().detach())

        # output the lowest distance cluster center
        minpos = dx_diff_list.index(min(dx_diff_list))
        reconstruct_output = cluster_combination_list[minpos]

        # check for accuracy
        # assert center_original in cluster_combination_list
        cate_original = cate_original[0].int()
        self.original.append(cate_original.tolist()) # to compute auc
        reconstruct_output = torch.tensor(reconstruct_output)
        self.recon.append(reconstruct_output.tolist()) # to compute auc
        # get the acc for each features
        acc = [int(i == j) for i, j in zip(reconstruct_output, cate_original)]
        
        return acc
    
    def get_dist_between_dx(self, dx_recon, dx_original):
        pdist = torch.nn.PairwiseDistance(p=2)
        pdistloss = torch.mean(pdist(dx_recon, dx_original))
        return pdistloss
    
def attack_cluster(testloader, 
                cluster_combination_list,
                server_model, 
                client_model, 
                loss_fn, 
                if_label_dp = False, 
                label_dp_prob = 0.9, 
                if_dp = False,
                if_save = False, 
                num_features=5,
    ):
    stats_output= {}
    acc_total = [0 for i in range(num_features)]
    num_samples = 0
    # initiaze the reconstruct machine
    recon_machine = Reconstruct_taobao(
        server_model = server_model,
        client_model = client_model
    )

    server_model.eval()
    client_model.eval()

    for batch_idx, batch in enumerate(testloader):
        num_samples += 1

        cate_original = batch['client']
        
        #print('cate original', cate_original.size())
        label = batch['label'].unsqueeze(1).to(torch.float32)

        # add label_dp if needed
        if if_label_dp:
            label, flipping = label_dp(label, prob=label_dp_prob)

        server_output_original = server_model(batch['server_categorical'],batch['server_continuous'])
        client_output_original = client_model(server_output_original,batch['client'])

        _, dx_original = client_model.backward(
                server_output_original, 
                client_output_original, 
                label,
                loss_fn, 
                update_model_grad=False)
        dx_original = dx_original[0]

        
        # adding DP stuff, average of test norm is 0.0304, if take half 
        if if_dp:
            dx_original, _ = dp(dx = dx_original, clip = 0.304/2, noise = 0.0005)

        # doing the attack
        acc = recon_machine.reconstruct(
                    cluster_combination_list,
                    server_output_original,
                    label,
                    dx_original,
                    cate_original,
        )
        
        acc_total = [sum(i) for i in zip(acc_total, acc)]  

        if batch_idx % 1000 == 0:
            print('batch idx', batch_idx)
            print('acc_total', acc_total)
            print('acc per features', [x / num_samples for x in acc_total])
            print('acc average over features', sum(acc_total)/(num_samples*num_features))
            
            # save the recon list
            if if_save:
                print(f'saving the attack results')
                name_o = 'attack/attack_original_dp' + str(batch_idx) + '.pkl'
                name_r = 'attack/attack_recon_dp' + str(batch_idx) + '.pkl'
                with open(name_o, 'wb') as f:
                    pickle.dump(recon_machine.original, f)
                with open(name_r, 'wb') as f:
                    pickle.dump(recon_machine.recon, f)
            
            if batch_idx > 0 :
                for i in range(num_features):
                    true = np.array(recon_machine.original)[:,i]
                    pred = np.array(recon_machine.recon)[:,i]
                    if i == 4:
                        #auc = metrics.roc_auc_score(true,pred)
                        f1_score = metrics.f1_score(true,pred)
                        recall_score = metrics.recall_score(true, pred)
                        precision = metrics.precision_score(true, pred)
                    else:
                        #auc = metrics.roc_auc_score(true, pred, multi_class='ovr')
                        f1_score = metrics.f1_score(true, pred, average='macro')
                        recall_score = metrics.recall_score(true, pred,  average='macro')
                        precision = metrics.precision_score(true, pred, average='macro')
                    print(f'recon on feature ({i}) f1_score = {f1_score}, recall={recall_score}, precision= {precision}')

    stats_output['acc_per_features'] = [x / num_samples for x in acc_total]
    stats_output['acc_avg'] = sum(acc_total)/(num_samples*num_features)
    print(stats_output['acc_avg'])

    return stats_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saving', type=int, required=False, default=0)
    parser.add_argument('--if_label_dp', type=bool, required=False, default=False)
    parser.add_argument('--if_dp', type=int, required=False, default=0)
    parser.add_argument('--label_dp_prob', type=float, required=False, default=0.9)
    args = parser.parse_args()
    if_label_dp = args.if_label_dp
    label_dp_prob = args.label_dp_prob
    if_dp = bool(args.if_dp)
    saving = bool(args.saving)
    print(f'starting attack cluster with')
    print(f'label_dp = {if_label_dp} and prob = {label_dp_prob}, DP = {if_dp}, and save = {saving}......')
    
    # prepare dataset
    tb_df = load_taobao_df()
    tb_train, tb_test = train_test_split(tb_df, test_size=0.2)
    #tb_train = tb_train[:len(tb_train)//100]
    tb_test = tb_test[:len(tb_test)//100]
    tb_test = tb_test.reset_index()
    #tb_train = tb_train.reset_index()
    print('finishing loading the data...')

    client_cols = ['cms_group_id','age_level','pvalue_level','shopping_level','occupation']
    embedding_cols = ['adgroup_id','cate_id','customer','brand','cms_segid'] # cms_segid is from user_related, should we change it???
    continuous = ['price']
    #trainset = Dataset_split(tb_train, embedding_cols, continuous, client_cols)
    testset = Dataset_split(tb_test, embedding_cols, continuous, client_cols)
    print(f'total number of test set to attack is {len(testset)}')

    test_dataloader_attack = DataLoader(testset, batch_size=1,shuffle=False)
    loss_fn = nn.BCELoss()
    # features can be attacked
    feature_offsite = ['cms_group_id','age_level','pvalue_level','shopping_level','occupation']
    offsite_category = [13, 7, 3, 3, 2] 
    cluster = []

    # create a combination
    for i in range(len(offsite_category)):
        temp = [j for j in range(offsite_category[i])]
        cluster.append(temp)

    cluster_combination_list = list(itertools.product(*cluster))
    print('total number of diffierent combination of category:', len(cluster_combination_list))

    # run the attack
    server_model = torch.load('model/server_model.pt')
    client_model = torch.load('model/client_model.pt')

    output = attack_cluster(
        test_dataloader_attack, 
        cluster_combination_list, 
        server_model,
        client_model, 
        loss_fn, 
        if_label_dp=if_label_dp, 
        label_dp_prob = label_dp_prob, 
        if_dp = if_dp,
        if_save = saving)
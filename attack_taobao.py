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
from collections import Counter

# client_features =  ['cms_segid','age_level','pvalue_level','shopping_level','occupation']
# 5 to attack

class Reconstruct_deepfm:
    def __init__(self, server_model, client_model):
        self.server_model = server_model
        self.client_model = client_model
        self.original = []
        self.recon = []

    def reconstruct(
        self,
        cluster_combination_list,
        server_output,
        label, #didnt flip through label dp
        dx_original,
        cate_original,
        loss_fn,
        majority_k = 1,
    ):
        self.client_model.eval()
        dx_diff_list = []
        for t in cluster_combination_list:
            # forward pass the client side model
            cate_try = t.clone().detach()[:-1]
            cate_try = cate_try.unsqueeze(0)
            label_try = t.clone().detach()[-1].float().unsqueeze(0).unsqueeze(0)
            
            client_output_recon = self.client_model(server_output,cate_try)

            # backward pass the client side model
            # get the gradient and compare with dx and store
            _, dx_recon = self.client_model.backward(
                            server_output,
                            client_output_recon, 
                            label_try, # using the trying label
                            loss_fn,
                            update_model_grad=False)
            
            dx_recon = dx_recon[0]
                    
            # get dist(dx, dx_original) and get smallest
            dx_diff = self.get_dist_between_dx(dx_recon, dx_original).detach()
            #dx_diff_list.append(dx_diff.clone().detach())
            dx_diff_list.append(dx_diff.item())

        # this part can do the majority vote
        dx_diff_np = np.array(dx_diff_list)
        minpos = np.argpartition(dx_diff_np,majority_k)[:majority_k].tolist()
        
        # output the lowest distance cluster center
        #minpos = dx_diff_list.index(min(dx_diff_list))
        #reconstruct_output = cluster_combination_list[minpos]
        reconstruct_output_k = []
        for v in range(majority_k):
            reconstruct_output_k.append(cluster_combination_list[minpos[v]].tolist())
        
        reconstruct_output = self.majority_vote(reconstruct_output_k, len(reconstruct_output_k[0]))
        # compute for accuracy
        label_original = label.squeeze().unsqueeze(0).int()
        cate_original = torch.cat((cate_original[0].int(),label_original),dim = 0)
        self.original.append(cate_original.tolist()) 
        self.recon.append(reconstruct_output) 
        return []


    def get_dist_between_dx(self, dx_recon, dx_original):
        pdist = torch.nn.PairwiseDistance(p=2)
        pdistloss = torch.mean(pdist(dx_recon, dx_original))
        return pdistloss

    def most_frequent(self,l):
        occurence_count = Counter(l)
        return occurence_count.most_common(1)[0][0]
    
    def majority_vote(self,k_list, num_features):
        res = []
        for i in range(num_features):
            vote = np.array(k_list)[:,i]
            vote = vote.tolist()
            #r = max(set(vote), key = vote.count)
            r = self.most_frequent(vote)
            res.append(r)
        return res


def attack_deepfm(testloader, 
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
                majority_k=1,
    ):
    stats_output= {}
    #acc_total = [0 for i in range(num_features)]
    num_samples = 0
    # initiaze the reconstruct machine
    recon_machine = Reconstruct_deepfm(
        server_model = server_model,
        client_model = client_model
    )
    if if_load:
        if use_dp:
            name_o = 'attack/nogroup/attack_original_dp' +str(noise)+'_'+ str(load_from) + '.pkl'
            name_r = 'attack/nogroup/attack_recon_dp' + str(noise)+'_'+ str(load_from) + '.pkl'
        elif use_label_dp:
            name_o = 'attack/nogroup/attack_original_labeldp' +str(label_dp_prob)+'_'+ str(load_from) + '.pkl'
            name_r = 'attack/nogroup/attack_recon_labeldp' + str(label_dp_prob)+'_'+ str(load_from) + '.pkl'
        else:
            name_o = 'attack/nogroup/attack_original_' + str(load_from)+ '.pkl'
            name_r = 'attack/nogroup/attack_recon_' + str(load_from)+ '.pkl'
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
    acc_total = 0
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
                majority_k=majority_k,
            )

            if num_samples % 10 == 0:
                print('num_samples', num_samples)
                
                # save the recon list
                if if_save:
                    print(f'saving the attack results')
                    if use_dp and use_label_dp:
                        name_o = 'attack/taobao/attack_original_dp' +str(noise)+'_'+ str(num_samples) + '_labeldp'+str(label_dp_prob)+'.pkl'
                        name_r = 'attack/taobao/attack_recon_dp'  +str(noise)+'_'+ str(num_samples) + '_labeldp'+str(label_dp_prob)+'.pkl'
                    elif use_dp and not use_label_dp:
                        name_o = 'attack/taobao/attack_original_dp' +str(noise)+'_'+ str(num_samples) + '.pkl'
                        name_r = 'attack/taobao/attack_recon_dp'  +str(noise)+'_'+ str(num_samples) + '.pkl'
                    elif use_label_dp and not use_dp:
                        name_o = 'attack/taobao/attack_original_labeldp' +str(label_dp_prob)+'_'+ str(num_samples) + '.pkl'
                        name_r = 'attack/taobao/attack_recon_labeldp'  +str(label_dp_prob)+'_'+ str(num_samples) + '.pkl'
                    elif fl:
                        name_o = 'attack/taobao/attack_original_fl'+str(num_samples)+'.pkl'
                        name_r = 'attack/taobao/attack_recon_fl'+str(num_samples)+'.pkl'
                    else:
                        name_o = 'attack/taobao/attack_original_'+str(num_samples)+'.pkl'
                        name_r = 'attack/taobao/attack_recon_'+str(num_samples)+'.pkl'
                    with open(name_o, 'wb') as f:
                        pickle.dump(recon_machine.original, f)
                    with open(name_r, 'wb') as f:
                        pickle.dump(recon_machine.recon, f)
            
                for i in range(num_features+1):
                    true = np.array(recon_machine.original)[:,i]
                    pred = np.array(recon_machine.recon)[:,i]
                    if i == (num_features-1) or i ==(num_features):
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
    parser.add_argument('--saving', type=int, required=False, default=0)
    parser.add_argument('--use_label_dp', type=bool, required=False, default=False)
    parser.add_argument('--use_dp', type=int, required=False, default=0)
    parser.add_argument('--label_dp_prob', type=float, required=False, default=0.1)
    parser.add_argument('--norm_clip', type=float, required=False)
    parser.add_argument('--noise', type=float, required=False)
    parser.add_argument('--loading', type=int, required=False, default=0)
    parser.add_argument('--load_from', type=int, required=False, default=0)
    parser.add_argument('--fl', type=int, required=False, default=0)
    parser.add_argument('--majority_k', type=int, required=False, default=1)
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
    majority_k = int(args.majority_k)
    #print(f'label_dp = {if_label_dp} and prob = {label_dp_prob}')
    print(f'Starting with...DP = {use_dp} with norm clip {norm_clip} and noise {noise}')
    print(f'label dp = {use_label_dp} with probability = {label_dp_prob}')
    print(f'save attack results = {saving}...and loading = {loading} from {load_from}...')
    
    if torch.cuda.is_available():
        device = "cuda"
        #print("USING CUDA")
    else:
        device = "cpu"
        #print("USING CPU")

    # prepare the dataset
    print('loading dataset')
    linear_feature_columns, dnn_feature_columns, client_feature_columns = get_columns()
    server_train,  client_train, y_train, server_test, client_test, y_test = load_taobao_df()

    # load the model
    print('loading model')
    if use_dp and use_label_dp:
        server_f = 'model/taobao/server_dp_0'+str(norm_clip)+str(noise)+'_labeldp'+str(label_dp_prob)+'.pt'
        client_f = 'model/taobao/client_dp_0'+str(norm_clip)+str(noise)+'_labeldp'+str(label_dp_prob)+'.pt'
    elif use_dp and not use_label_dp:    
        server_f = 'model/early/server_model_dp_0'+str(norm_clip)+str(noise)+'_nogroup.pt'
        client_f = 'model/early/client_model_dp_0'+str(norm_clip)+str(noise)+'_nogroup.pt'
    elif fl:
        server_f = 'model/fl/server_model_0_nogroup_fl.pt'
        client_f = 'model/fl/client_model_0_nogroup_fl.pt'
    elif use_label_dp and not use_dp:
        server_f = 'model/early/server_model_labeldp_0'+str(label_dp_prob)+'_nogroup.pt'
        client_f = 'model/early/client_model_labeldp_0'+str(label_dp_prob)+'_nogroup.pt'
    else:
        server_f = 'model/early/server_model_0_nogroup.pt'
        client_f = 'model/early/client_model_0_nogroup.pt'
    
    print(server_f)
    print(client_f)
    server_model = torch.load(server_f)
    client_model = torch.load(client_f)
    
    # load testloader with batch size = 1
    print('load testloader')
    test_loader, _ = data_to_dataloader(server_model, server_test, client_test, y_test, 1, True)
    
    # evaluate on the model first
    #print('evaluting the model on testset first')
    #test_loader_eval, _ = data_to_dataloader(server_model, server_test, client_test, y_test, 1024, True)
    #eval_result = evaluate_split_deepfm(test_loader_eval, y_test, server_model, client_model, device, epoch=0, verbose = 2)

    # create a combination for attack to go through
    #feature_offsite = ['cms_group_id','age_level','pvalue_level','shopping_level','occupation','label']
    #offsite_category = [13, 7, 3, 3, 2, 2]
    feature_offsite = ['age_level','pvalue_level','shopping_level','occupation','label']
    num_features = len(feature_offsite) -1
    offsite_category = [7, 3, 3, 2, 2]
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
    output = attack_deepfm(
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
        num_features=num_features,
        majority_k=majority_k)


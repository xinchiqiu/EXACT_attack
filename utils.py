from sklearn.metrics import log_loss, roc_auc_score
#from sklearn.model_selection import StratifiedKFold
#from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import itertools
from sklearn import metrics
#from sklearn.cluster import KMeans
#from sklearn.neighbors import KNeighborsClassifier

from DeepCTR.deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat, get_feature_names)

def load_taobao_df():
    ad = "/datasets/taobao/ad_feature.csv"
    ad_feature_df =  pd.read_csv(ad)

    raw_sample = "/datasets/taobao/raw_sample.csv"
    raw_sample_df = pd.read_csv(raw_sample)

    user = "/datasets/taobao/user_profile.csv"
    user_profile_df = pd.read_csv(user)

    # memory optimize for ad feature dataframe
    optimized_gl = raw_sample_df.copy()

    gl_int = raw_sample_df.select_dtypes(include=['int'])
    converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')
    optimized_gl[converted_int.columns] = converted_int


    gl_obj = raw_sample_df.select_dtypes(include=['object']).copy()
    converted_obj = pd.DataFrame()
    for col in gl_obj.columns:
        num_unique_values = len(gl_obj[col].unique())
        num_total_values = len(gl_obj[col])
        if num_unique_values / num_total_values < 0.5:
            converted_obj.loc[:,col] = gl_obj[col].astype('category')
        else:
            converted_obj.loc[:,col] = gl_obj[col]
    optimized_gl[converted_obj.columns] = converted_obj
    raw_sample_df = optimized_gl.copy()
    raw_sample_df_new = raw_sample_df.rename(columns = {"user": "userid"})
    optimized_g2 = ad_feature_df.copy()
    g2_int = ad_feature_df.select_dtypes(include=['int'])
    converted_int = g2_int.apply(pd.to_numeric,downcast='unsigned')
    optimized_g2[converted_int.columns] = converted_int

    g2_float = ad_feature_df.select_dtypes(include=['float'])
    converted_float = g2_float.apply(pd.to_numeric,downcast='float')
    optimized_g2[converted_float.columns] = converted_float

    optimized_g3 = user_profile_df.copy()

    g3_int = user_profile_df.select_dtypes(include=['int'])
    converted_int = g3_int.apply(pd.to_numeric,downcast='unsigned')
    optimized_g3[converted_int.columns] = converted_int

    g3_float = user_profile_df.select_dtypes(include=['float'])
    converted_float = g3_float.apply(pd.to_numeric,downcast='float')
    optimized_g3[converted_float.columns] = converted_float

    # combine 3 tables
    df1 = raw_sample_df_new.merge(optimized_g3, on="userid")
    final_df = df1.merge(optimized_g2, on="adgroup_id")
    
    # make it faster, need to remove
    #final_df = final_df[:300000]
    
    final_df['pvalue_level'] = final_df['pvalue_level'].fillna(2, )
    final_df['final_gender_code'] = final_df['final_gender_code'].fillna(1, )
    final_df['age_level'] = final_df['age_level'].fillna(3, )
    final_df['shopping_level'] = final_df['shopping_level'].fillna(2, )
    final_df['occupation'] = final_df['occupation'].fillna(0, )
    final_df['brand'] = final_df['brand'].fillna(0, )
    final_df['customer'] = final_df['customer'].fillna(0, )
    final_df['cms_group_id'] = final_df['cms_group_id'].fillna(13, )


    final_df['pvalue_level'] -= 1
    final_df['shopping_level'] -= 1
    final_df = final_df.astype({"cms_segid": int, 
                                "cms_group_id": int, 
                                'clk': int, 
                                'adgroup_id': int, 
                                'final_gender_code':int,
                                'age_level':int,
                                'pvalue_level':int,
                                'shopping_level':int,
                                'occupation':int,
                                'cate_id':int,
                                'customer':int,
                                'brand':int}
                                )
    
    #sparse_features = ['adgroup_id','cate_id','customer','brand','cms_segid', 'final_gender_code','userid'] # for non_split
    #sparse_features = ['adgroup_id', 'cate_id', 'customer', 'brand', 'cms_segid', 'cms_group_id','userid'] #for non-split
    
    sparse_features = ['adgroup_id','cate_id','customer','brand','userid']
    final_df[sparse_features] = final_df[sparse_features].fillna('-1', )    
    dense_features = ['price']
    #client_features =  ['cms_segid', 'final_gender_code','age_level','pvalue_level','shopping_level','occupation']
    client_features =  ['age_level','pvalue_level','shopping_level','occupation']
    target = ['clk']
    
    mms = MinMaxScaler(feature_range=(0, 1))
    final_df[dense_features] = mms.fit_transform(final_df[dense_features])
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in sparse_features] + [DenseFeat(feat, 1, )for feat in dense_features]# + [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8), 1) for feat in sequence_features] 
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )

    #client_feature_columns = [DenseFeat(feat, 1, )for feat in client_features]
    client_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in client_features]
    client_feature_names = get_feature_names(client_feature_columns, )
    train, test = train_test_split(final_df, test_size=0.1, random_state = 42)

    #train_other, train = train_test_split(train, test_size=0.003, random_state = 42)
    #test_other, test = train_test_split(test, test_size=0.003, random_state = 42)
    # print(len(train))
    #test = test[:5000]
    #print(len(train))

    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    train_client_input = {name:train[name] for name in client_feature_names}
    test_client_input = {name:test[name] for name in client_feature_names}

    return train_model_input,  train_client_input, train[target].values, test_model_input, test_client_input,test[target].values

def get_columns():
    #sparse_features = ['userid', 'adgroup_id', 'final_gender_code', 'cate_id']
    #sparse_features = ['adgroup_id','cate_id','customer','brand','cms_segid', 'final_gender_code','userid'] #first try
    sparse_features = ['adgroup_id','cate_id','customer','brand','userid']    
    dense_features = ['price']
    #client_features =  ['cms_segid', 'final_gender_code','age_level','pvalue_level','shopping_level','occupation']
    client_features =  ['age_level','pvalue_level','shopping_level','occupation']
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in sparse_features] + [DenseFeat(feat, 1, )for feat in dense_features]# + [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8), 1) for feat in sequence_features] 
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    client_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in client_features]
    return linear_feature_columns, dnn_feature_columns, client_feature_columns

class Dataset_split(Dataset):
    def __init__(self, X, emb_cols, server_continuous_cols, non_emb_cols):
        X = X.copy()
        self.server_categorical = X.loc[:,emb_cols].copy().values.astype(np.int64) #categorical columns
        self.client = X.reindex(columns = non_emb_cols).values.astype(np.float32)
        self.server_continuous = X.reindex(columns = server_continuous_cols).values.astype(np.float32)
        self.y = X['clk']
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        #return self.server_categorical[idx], self.server_continuous[idx], self.client[idx], self.y[idx]

        return {'server_categorical': self.server_categorical[idx],
                'server_continuous': self.server_continuous[idx],
                'client': self.client[idx],
                'label': self.y[idx]}

# server side arch
class server_arch(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb, self.n_cont = 250, n_cont
        self.emb_drop = nn.Dropout(0.6)
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 64)
        

    def forward(self, x_server_categorical,x_server_continuous):
        x = [e(x_server_categorical[:,i]) for i,e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = torch.cat([x,x_server_continuous], 1)
        x = self.emb_drop(x)
        x = F.relu(self.lin1(x))
        return x

class client_arch(nn.Module):
    def __init__(self, n_server_output, n_client_input):
        super().__init__()
        self.n_server_output = n_server_output
        self.n_client_input = n_client_input
        self.lin1 = nn.Linear(self.n_server_output + self.n_client_input, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 1)
        #self.bn1 = nn.BatchNorm1d(self.n_cont)
        #self.bn2 = nn.BatchNorm1d(64)
        #self.bn3 = nn.BatchNorm1d(32)
        #self.emb_drop = nn.Dropout(0.6)
        #self.drops = nn.Dropout(0.3)
        

    def forward(self, server_output, x_client):
        cut_layer = torch.cat([server_output, x_client], 1)
        x = F.relu(self.lin1(cut_layer))
        #x = self.drops(x)
        #x = self.bn2(x)
        x = F.relu(self.lin2(x))
        #x = self.drops(x)
        #x = self.bn3(x)
        x = torch.sigmoid(self.lin3(x))
        return x

    def set_model_gradients(self, gradient):
        params = list(self.parameters())
        for p, grad in zip(params, gradient):
            p.grad = grad.clone()

    def backward(self, cut_node, output, target, loss_fn, update_model_grad=True):
        loss = loss_fn(output, target)
        params = list(self.parameters())
                
        # gradients dE/dw_n
        de_dw = torch.autograd.grad(
            [loss],
            params,
            retain_graph = True,
        )

        if update_model_grad:
            self.set_model_gradients(de_dw)


        de_dx = torch.autograd.grad(
            [loss],
            cut_node,
            retain_graph = True,
        )

        return de_dw, de_dx

def weighted_binary_cross_entropy(output, target, weights=None):
        
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def validate_split(server_model, client_model, loss_fn, test_dataloader, device):
    server_model.eval()
    client_model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    y_proba_list = []
    y_true_list = []
    y_pred_list = []
    for batch in test_dataloader:
        for key, value in batch.items():
                batch[key] = batch[key].to(device)

        y = batch['label'].unsqueeze(1).to(torch.float32)

        current_batch_size = y.shape[0]

        server_output = server_model(batch['server_categorical'], batch['server_continuous'])
        out = client_model(server_output, batch['client'])
        
        y_proba_list.extend(out.detach().cpu().numpy())
        y_true_list.extend(y.detach().cpu().numpy())

        loss = loss_fn(out, y)
        sum_loss += current_batch_size*(loss.item())
        total += current_batch_size
        pred = torch.round(out).squeeze()

        y_pred_list.extend(pred.detach().cpu().numpy())
        y = y.squeeze()
        correct += (pred == y).sum()
        

    print("test loss %.3f and accuracy %.3f" % (sum_loss/total, correct/total))
    auc = metrics.roc_auc_score(y_true_list,y_proba_list)
    print("roc_auc_score=",auc)
    conf_matrix = metrics.confusion_matrix(y_true_list,y_pred_list)
    print('confusion matrix=',conf_matrix)
    return sum_loss/total, correct/total


# build training loop that for separate model
def one_round_split_train(server_model, client_model, server_opt, client_opt, batch , loss_fn, if_dp = False):
    # zero all optimizers
    server_opt.zero_grad()
    client_opt.zero_grad()

    server_output = server_model(batch['server_categorical'],batch['server_continuous'])
    client_output = client_model(server_output,batch['client'])
    label = batch['label'].unsqueeze(1).to(torch.float32)

    loss = loss_fn(client_output, label)
    #loss = weighted_binary_cross_entropy(client_output, label, weights=class_weight)
    #loss = class_weight[1] * (label * torch.log(client_output)) + class_weight[0] * ((1 - label) * torch.log(1 - client_output))

    # backward
    dw, dx = client_model.backward(server_output, client_output, label,loss_fn, update_model_grad=True)

    # adding the DP on dx
    if if_dp:
        dx_dp, _  = dp(dx = dx[0], clip = 0.304/2, noise = 0.0005)
        server_output.backward(dx_dp)
    else:
        server_output.backward(dx)

    server_opt.step()
    client_opt.step()
    return loss, client_output     

def fit(server_model, client_model,loss_fn, trainloader, testloader, epochs, device, lr= 1, momentum=0, if_dp=False):

    total_loss = 0
    total = 0
    print('starting fiting')
    for e in range(0,epochs):
        print('epoch', e)
        for batch in trainloader:
            #if e > 5:
            #    lr = lr/2
            server_opt = torch.optim.SGD(server_model.parameters(),lr=lr, momentum = momentum)
            client_opt = torch.optim.SGD(client_model.parameters(),lr=lr, momentum = momentum)
            
            for key, value in batch.items():
                batch[key] = batch[key].to(device)

            _, output = one_round_split_train(server_model, client_model, server_opt, client_opt, batch, loss_fn, if_dp)
            #total_loss += loss
            total += batch['label'].shape[0]

            #if batch_idx % 500 == 0:
        val_loss, val_acc = validate_split(server_model, client_model, loss_fn, testloader, device)
        print(f'saving for epoch {e}')
        server_f = 'model/server_model_'+str(e)+'.pt'
        client_f = 'model/client_model_'+str(e)+'.pt'
        torch.save(server_model, server_f)
        torch.save(client_model, client_f)

def label_dp(label, prob):
    size = len(label)
    flipping = [random.random() < prob for _ in range(size)]
    new_label = torch.tensor([x+y-2*x*y for x,y in zip(label, flipping)])
    return new_label.unsqueeze(1).to(torch.float32), flipping

def dp(dx, clip, noise, device):
    # first need to clip
    norm = torch.norm(dx, p=2)
    dx_clip = dx / max(1, norm/clip)
    # then add the noise
    dx_noise = dx_clip + torch.normal(mean = 0, std = clip* noise, size = dx.size()).to(device)
    return dx_noise, norm.cpu()


def fedavg_dw(dw):
    num_client = len(dw)
    l = len(dw[0][0])
    dw_avg = []
    for idx in range(l):
        dw_list = [dw[i][idx].cpu().detach() for i in range(num_client)]
        avg = torch.mean(torch.stack(dw_list))
        dw_avg.append(avg)
    return dw_avg

def categorical_values_to_int(df, columns,given_dicts):
    # san check
    assert set(df.columns).issuperset(columns)

    def fn(col):
        cat = dict([(o, i) for i, o in enumerate(set(df[col]))])
        for d in given_dicts:
            if set(d.keys()) == set(cat.keys()):
                cat = d
                break
        df[col] = df[col].apply(lambda x: cat[x])
        return cat

    return df, [fn(c) for c in columns]


def load_banking():
    bank_path = '/datasets/bank/bank_full.csv'
    bank_df = pd.read_csv(bank_path,sep=";")
    bank_df['pdays'] = bank_df['pdays'].apply(lambda x: 999 if x == -1 else x)
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    dicts = categorical_values_to_int(bank_df, columns=cat_cols, given_dicts=[{'no': 0, 'yes': 1}])[1]
    col_mappings = dict(zip(cat_cols, dicts))
    for col in ['age', 'balance', 'campaign', 'previous', 'pdays']:
        arr = bank_df[col].values
        min_arr, max_arr = arr.min(), arr.max()
        bank_df[col] = (arr - arr.min()) / (arr.max() - arr.min())
        col_mappings[col] = (arr.min(), arr.max())

    client_features = ['marital', 'job', 'education', 'housing','loan','contact']
    dense_features = ['balance','age']
    sparse_features = ['day','month','duration','campaign','pdays','previous','poutcome']
    target = ['y']


    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in sparse_features] + [DenseFeat(feat, 1, )for feat in dense_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )
    client_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in client_features]
    client_feature_names = get_feature_names(client_feature_columns, )

    train, test = train_test_split(bank_df, test_size=0.1, random_state = 42)
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    train_client_input = {name:train[name] for name in client_feature_names}
    test_client_input = {name:test[name] for name in client_feature_names}

    return train_model_input,  train_client_input, train[target].values, test_model_input, test_client_input,test[target].values

def get_banking_columns():
    sparse_features = ['day','month','duration','campaign','pdays','previous','poutcome']  
    dense_features = ['balance','age']
    client_features = ['marital', 'job', 'education', 'housing','loan','contact']

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in sparse_features] + [DenseFeat(feat, 1, )for feat in dense_features]# + [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8), 1) for feat in sequence_features] 
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    client_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in client_features]
    return linear_feature_columns, dnn_feature_columns, client_feature_columns


def load_adult(client_features, sparse_features, dense_features):
    #adult = '/datasets/adult/adult.csv'
    adult = 'adult.csv'
    df = pd.read_csv(adult,sep=",")
    for col in df.columns:
        if "?" in df[col].unique().tolist():
            df[col] = df[col].apply(lambda x: 'unknown' if x == '?' else x)
        
    cat_cols = ['workclass', 'educational-num', 'marital-status', 'occupation',
        'relationship', 'race', 'gender', 'native-country']
    num_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    label_col = ['income']
    df.reset_index(drop=True, inplace=True)
    ids = ['N%05d' % i for i in range(len(df))]
    df.insert(0, 'ID', ids)
    df = df.loc[:, ['ID'] + cat_cols + num_cols + label_col]
    df.set_index('ID', inplace=True)
    for col in cat_cols + label_col:
        lst = sorted(df[col].unique().tolist())
        mapping = dict(zip(lst, range(len(lst))))
        df[col] = df[col].apply(lambda x: mapping[x])
    for col in num_cols:
        arr = df[col].values
        df[col] = (arr - arr.min()) / (arr.max() - arr.min())
    df = df.rename(columns={"educational-num": "education", 
                            "marital-status": "marital", 
                            "native-country":"country", 
                            "capital-gain": "gain", 
                            "capital-loss": "loss", 
                            "hours-per-week":"hours_per_week"})

    # prepare the column for offsite features and model
    #client_features = ['gender', 'race', 'relationship', 'marital']
    # dense_features = ['age','gain', 'loss', 'hours_per_week']
    #sparse_features = ['workclass','education','occupation','country']
    # client_features = ['gender','race','marital','relationship','occupation']
    # sparse_features = ['country','workclass','education']
    target = ['income']
    
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in sparse_features] + [DenseFeat(feat, 1, )for feat in dense_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )
    client_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in client_features]
    client_feature_names = get_feature_names(client_feature_columns, )

    train, test = train_test_split(df, test_size=0.1, random_state = 2)
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    train_client_input = {name:train[name] for name in client_feature_names}
    test_client_input = {name:test[name] for name in client_feature_names}

    return train_model_input,  train_client_input, train[target].values, test_model_input, test_client_input,test[target].values

def get_adult_columns(client_features, sparse_features, dense_features):
    #client_features = ['gender', 'race', 'relationship', 'marital']
    # client_features = ['gender','race','marital','relationship','occupation']
    # sparse_features = ['country','workclass','education']
    # dense_features = ['age','gain', 'loss', 'hours_per_week']
    #sparse_features = ['workclass','education','occupation','country'] 

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in sparse_features] + [DenseFeat(feat, 1, )for feat in dense_features]# + [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8), 1) for feat in sequence_features] 
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    client_feature_columns = [SparseFeat(feat, vocabulary_size=2000000,embedding_dim=8) for feat in client_features]
    return linear_feature_columns, dnn_feature_columns, client_feature_columns
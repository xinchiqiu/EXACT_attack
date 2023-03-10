from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
#import pandas_profiling 
from sklearn.metrics import log_loss, roc_auc_score
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

def knn_original(tb_train, tb_test, server_cols, offsite_cols, n_neighbors):
    # doing the knn prediction to recon the offsite features
    # using the original server_features
    print(f'Runnign knn on original server features with n_neighbors = {n_neighbors}..')
    X_train = tb_train.loc[:,server_cols].copy().values.astype(np.int64) 
    y_train = tb_train.loc[:,offsite_cols].copy().values.astype(np.int64)
    X_test = tb_test.loc[:,server_cols].copy().values.astype(np.int64) 
    y_test = tb_test.loc[:,offsite_cols].copy().values.astype(np.int64)
    #X_train, X_test, y_train, y_test = train_test_split(server_data, offsite_data, test_size=0.2)

    for i in range(len(offsite_cols)):
        train_y = y_train[:,i]
        test_y = y_test[:,i]
        knn =  KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train,train_y)
        y_pred = knn.predict(X_test)
        accuracy = metrics.accuracy_score(test_y, y_pred)
        if offsite_cols[i] == 'occupation':
            auc = metrics.roc_auc_score(test_y, knn.predict_proba(X_test)[:, 1])
            f1_score = metrics.f1_score(test_y, knn.predict(X_test))
            recall_score = metrics.recall_score(test_y, knn.predict(X_test))
            precision = metrics.precision_score(test_y, knn.predict(X_test))
        else:
            auc = metrics.roc_auc_score(test_y, knn.predict_proba(X_test), multi_class='ovr')
            f1_score = metrics.f1_score(test_y, knn.predict(X_test), average='macro')
            recall_score = metrics.recall_score(test_y, knn.predict(X_test),  average='macro')
            precision = metrics.precision_score(test_y, knn.predict(X_test), average='macro')
        print(f'recon on feature ({offsite_cols[i]}), acc = {accuracy}, roc_auc = {auc}, f1_score = {f1_score},recall={recall_score}, precision= {precision} ')


# then kmeans the server features first, then do the knn
def knn_kmeans(tb_train, tb_test, server_cols, offsite_cols, n_cluster_kmeans=100, n_neighbors=30):
    print(f'running knn on kmeans of server features with n_cluster_kmeans = {n_cluster_kmeans}, n_neighjbors = {n_neighbors}.')
    # doing the kmeans of server features first
    X_train = tb_train.loc[:,server_cols].copy().values.astype(np.int64) 
    y_train = tb_train.loc[:,offsite_cols].copy().values.astype(np.int64)
    X_test = tb_test.loc[:,server_cols].copy().values.astype(np.int64) 
    y_test = tb_test.loc[:,offsite_cols].copy().values.astype(np.int64)
    #X_train, X_test, y_train, y_test = train_test_split(server_data, offsite_data, test_size=0.2)
    kmeans = KMeans(n_clusters=n_cluster_kmeans) #takes 20min
    kmeans.fit(X_train)
    x_train_kmeans = kmeans.predict(X_train)
    x_test_kmeans = kmeans.predict(X_test)
    x_train_center = [kmeans.cluster_centers_[i] for i in x_train_kmeans]
    x_test_center = [kmeans.cluster_centers_[i] for i in x_test_kmeans]
    print('finish kmeans')

    #start knn
    for i in range(len(offsite_cols)):
        train_y = y_train[:,i]
        test_y = y_test[:,i]
        knn =  KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(x_train_center,train_y)
        y_pred = knn.predict(x_test_center)
        accuracy = metrics.accuracy_score(test_y, y_pred)
        print(offsite_cols[i])
        if offsite_cols[i] == 'occupation':
            auc = metrics.roc_auc_score(test_y, knn.predict_proba(X_test)[:, 1])
            f1_score = metrics.f1_score(test_y, knn.predict(X_test))
        else:
            auc = metrics.roc_auc_score(test_y, knn.predict_proba(X_test), multi_class='ovr')
            f1_score = metrics.f1_score(test_y, knn.predict(X_test), average='macro')

        print(f'recon on feature ({offsite_cols[i]}_, acc = {accuracy}, roc_auc = {auc}, f1_score = {f1_score} ')

def knn_serveroutput(tb_train, tb_test, offsite_cols, server_model, n_neighbors=30):
    # doing the kmeans of server output
    # get the server output first
    print(f'running knn on server_output with n_neighjbors = {n_neighbors}.')
    embedding_cols = ['adgroup_id','cate_id','customer','brand','cms_segid'] # cms_segid is from user_related, should we change it???
    continuous = ['price']
    trainset = Dataset_split(tb_train, embedding_cols, continuous, offsite_cols)
    testset = Dataset_split(tb_test, embedding_cols, continuous, offsite_cols)
    train_dataloader = DataLoader(trainset, batch_size=32,shuffle=False)
    test_dataloader = DataLoader(testset, batch_size=32,shuffle=False)
    server_model.eval()
    server_output_train = None
    server_output_test = None
    y_train = None
    y_test = None

    for batch in train_dataloader:
        for key, value in batch.items():
                batch[key] = batch[key].to(device)
        if server_output_train is None:
            server_output_train = server_model(batch['server_categorical'],batch['server_continuous']).detach().numpy()
            y_train = batch['client'].detach().numpy()
        else:
            out = server_model(batch['server_categorical'],batch['server_continuous']).detach().numpy()
            server_output_train = np.concatenate((server_output_train,out), axis = 0)
            y_train = np.concatenate((y_train,batch['client'].numpy() ), axis = 0)

    for batch in test_dataloader:
        for key, value in batch.items():
                batch[key] = batch[key].to(device)
        if server_output_test is None:
            server_output_test = server_model(batch['server_categorical'],batch['server_continuous']).detach().numpy()
            y_test = batch['client'].detach().numpy()
        else:
            out = server_model(batch['server_categorical'],batch['server_continuous']).detach().numpy()
            server_output_test = np.concatenate((server_output_test,out), axis = 0)
            y_test = np.concatenate((y_test,batch['client'].detach().numpy()), axis = 0)

    #run knn on server_output 
    for i in range(len(offsite_cols)):
        if i == 4:
            train_y = y_train[:,i]
            test_y = y_test[:,i]
            knn =  KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(server_output_train,train_y)
            y_pred = knn.predict(server_output_test)
            accuracy = metrics.accuracy_score(test_y, y_pred)
            if offsite_cols[i] == 'occupation':
                auc = metrics.roc_auc_score(test_y, knn.predict_proba(server_output_test)[:, 1])
                f1_score = metrics.f1_score(test_y, knn.predict(server_output_test))
                recall_score = metrics.recall_score(test_y, knn.predict(server_output_test))
                precision = metrics.precision_score(test_y, knn.predict(server_output_test))
            else:
                auc = metrics.roc_auc_score(test_y, knn.predict_proba(server_output_test), multi_class='ovr')
                f1_score = metrics.f1_score(test_y, knn.predict(server_output_test), average='macro')
                recall_score = metrics.recall_score(test_y, knn.predict(server_output_test),  average='macro')
                precision = metrics.precision_score(test_y, knn.predict(server_output_test), average='macro')
            print(f'recon on feature ({offsite_cols[i]}), acc = {accuracy}, roc_auc = {auc}, f1_score = {f1_score}, recall={recall_score}, precision= {precision} ')

if __name__  == '__main__':
    # device
    device = 'cpu'

    # prepare dataset
    tb_df = load_taobao_df()
    tb_train, tb_test = train_test_split(tb_df, test_size=0.2)
    tb_train = tb_train[:len(tb_train)//100]
    tb_test = tb_test[:len(tb_test)//100]
    tb_test = tb_test.reset_index()
    tb_train = tb_train.reset_index()
    print('finishing loading the data...')

    # knn original 
    server_cols = ['adgroup_id','cate_id','customer','brand','cms_segid']
    feature_offsite = ['cms_group_id','age_level','pvalue_level','shopping_level','occupation']
    #knn_original(tb_train = tb_train, tb_test = tb_test, server_cols=server_cols, offsite_cols=feature_offsite, n_neighbors=30)

    embedding_sizes = [(846811+1, 50), (12960+1,50), (255875+1,50),(461497+1,50),(96+1,50)]
    # if not train anything
    #server_model = server_arch(embedding_sizes, 1).to(device) 
    server_model = torch.load('model/server_model.pt')
    feature_offsite = ['cms_group_id','age_level','pvalue_level','shopping_level','occupation']
    knn_serveroutput(tb_train = tb_train, tb_test = tb_test, offsite_cols = feature_offsite, server_model = server_model, n_neighbors=30)

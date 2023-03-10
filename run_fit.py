import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import itertools
from sklearn import metrics
import argparse
from utils import *



print('starting')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--momentum', type=float, required=True)
args = parser.parse_args()
epochs = args.epochs
lr = args.lr
momentum = args.momentum
batch_size = args.batch_size
print(f'Training for {epochs} epochs, with batch size {batch_size}, lr={lr} and momentum={momentum}')

tb_df = load_taobao_df()
print(f'finishing loading datasets')

tb_train, tb_test = train_test_split(tb_df, test_size=0.2)
tb_train = tb_train[:len(tb_train)//10]
tb_test = tb_test[:len(tb_test)//10]
#tb_train = tb_train[:1000]
#tb_test = tb_test[:1000]
tb_test = tb_test.reset_index()
tb_train = tb_train.reset_index()
print(f'finishing preparing datasets')

client_cols = ['cms_group_id','age_level','pvalue_level','shopping_level','occupation']
embedding_cols = ['adgroup_id','cate_id','customer','brand','cms_segid'] # cms_segid is from user_related, should we change it???
continuous = ['price']
trainset = Dataset_split(tb_train, embedding_cols, continuous, client_cols)
testset = Dataset_split(tb_test, embedding_cols, continuous, client_cols)
print('finish prepare trainset.')

# device
if torch.cuda.is_available():
    device = "cuda"
    print("USING CUDA")
else:
    device = "cpu"
    print("USING CPU")

# sampler
counts = [sum(tb_train['clk']), len(tb_train) - sum(tb_train['clk'])]
class_weights = [sum(counts)/c for c in counts]
example_weights = [1/class_weights[e] for e in tb_train['clk']]
sampler = WeightedRandomSampler(example_weights, len(example_weights),replacement=True)

train_dataloader = DataLoader(trainset, batch_size=batch_size,sampler=sampler)
test_dataloader = DataLoader(testset, batch_size=batch_size,shuffle=True)


# fitting
print(f'starting fitting')
loss_fn = nn.BCELoss()
device = 'cpu'
embedded_cols = {'adgroup_id':846811,
            'cate_id': 12960,
            'customer': 255875,
            'brand': 461497,
            'cms_segid': 96}
embedding_sizes = [(846811+1, 50), (12960+1,50), (255875+1,50),(461497+1,50),(96+1,50)]
client_cols = ['cms_group_id','age_level','pvalue_level','shopping_level','occupation']
server_continous = ['price']
server_model = server_arch(embedding_sizes, 1).to(device)
client_model = client_arch(64, len(client_cols)).to(device)

fit(server_model, client_model, loss_fn, train_dataloader, test_dataloader, epochs, device, class_weights, lr, momentum)

torch.save(server_model, 'model/server_model_10.pt')
torch.save(client_model, 'model/client_model_10.pt')
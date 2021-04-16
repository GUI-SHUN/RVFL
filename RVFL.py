
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib.request
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_train = np.loadtxt('yeast_train_data.txt')
data_test = np.loadtxt('yeast_test_data.txt')
label_train = np.loadtxt('yeast_train_label.txt')
label_test = np.loadtxt('yeast_test_label.txt')
#data = dataset_raw[:, 0:-1]
#label = dataset_raw[:, -1]
INPUT_SIZE = data_train.shape[1]
OUTPUT_SIZE = int(np.max(label_train)+1)

#data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.4, random_state=0)
data_train = torch.from_numpy(data_train).float()
data_test = torch.from_numpy(data_test).float()
label_train = torch.from_numpy(label_train).long()
label_test = torch.from_numpy(label_test).long()


HIDDEN = 64
LAMBDA = 0.9

class RVFL(nn.Module):
    def __init__(self):
        super(RVFL, self).__init__()
        self.linear_dir = nn.Linear(INPUT_SIZE, OUTPUT_SIZE, bias=True)
        self.linear_hidden = nn.Linear(INPUT_SIZE, HIDDEN, bias=True)
        self.linear_hidden_out = nn.Linear(HIDDEN, OUTPUT_SIZE, bias=True)
        self.act_hidden = nn.Sigmoid()
        
    def forward(self, x):
        dir_data = self.linear_dir(x)
        hidden_data = self.act_hidden(self.linear_hidden(x))
        feature = torch.cat((hidden_data, x), -1)
        hidden_out = self.linear_hidden_out(hidden_data)
        return feature, dir_data+hidden_out

criterion = nn.CrossEntropyLoss()
Y = torch.zeros(label_train.shape[0], OUTPUT_SIZE).scatter_(1, label_train.reshape(-1, 1), 1)

rvfl = RVFL()
rvfl.eval()
out_feature, _ = rvfl(data_train)
weight = torch.mm(torch.mm(torch.inverse(LAMBDA*torch.eye(out_feature.shape[1], out_feature.shape[1]) + torch.mm(out_feature.t(), out_feature)), out_feature.t()), Y)
weight = weight.t()

rvfl.state_dict()['linear_hidden_out.weight'].copy_(weight[:,0:HIDDEN])
rvfl.state_dict()['linear_dir.weight'].copy_(weight[:,HIDDEN:])

_, pred_train = rvfl(data_train)
loss_train = criterion(pred_train, label_train)
pred_train = torch.max(pred_train, 1)[1]
correct_train = torch.sum(pred_train==label_train).item()
print('Accuracy_train: %f, Loss_train: %f' % (correct_train/label_train.shape[0], loss_train))

_, pred = rvfl(data_test)
loss_pred = criterion(pred, label_test)
pred = torch.max(pred, 1)[1]
correct = torch.sum(pred == label_test).item()
print('Accuracy: %f, Loss: %f' % (correct/label_test.shape[0], loss_pred))

            














import pandas as pd  
import numpy as np  
from time import time
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
from torchvision import transforms, utils
torch.manual_seed(1)

EPOCH = 100              # train the training data n times
BATCH_SIZE = 64
TIME_STEP = None    # rnn time step
INPUT_SIZE = 78         # rnn input size
LR = 0.01               # learning rate

class CustomDataset(Dataset):
  def __init__(self, features, labels):
    self.features = features
    self.labels = labels
  def __len__(self):
    return(len(self.labels))
  def __getitem__(self, index):
    return self.features[index],self.labels[index]

#dataroot="~/home/szp/guyi/CICIDS2017.csv"
dataroot=r"E:\dataset\MachineLearningCVE\CICIDS2017.csv"  
df=pd.read_csv(dataroot,header=0,low_memory=False)
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
df.iloc[:,0:-1]=scaler.fit_transform(df.iloc[:,0:-1])
x_train, x_test, y_train,y_test = train_test_split(df.iloc[:,0:-1].as_matrix(), df[' Label'].as_matrix(), test_size=0.3, random_state=42)
train_data = CustomDataset(x_train, y_train)
test_data = CustomDataset(x_test, y_test)
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class RNN(nn.Module):
  def __init__(self):
    super(RNN, self).__init__()
    self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
      input_size=INPUT_SIZE,
      hidden_size=32,         # rnn hidden unit
      num_layers=2,           # number of rnn layer
      batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
      )
    self.out = nn.Linear(32, 8)
  def forward(self, x):
    # x shape (batch, time_step, input_size)
    # r_out shape (batch, time_step, output_size)
    # h_n shape (n_layers, batch, hidden_size)
    # h_c shape (n_layers, batch, hidden_size)
    r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
    # choose r_out at the last time step
    out = self.out(r_out[:, -1, :])
    return out

rnn = RNN()
print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted


for epoch in range(EPOCH):
  for step, (b_x, b_y) in enumerate(trainloader):   # gives batch data
    b_x = b_x.view(-1, 1, INPUT_SIZE).float()       # reshape x to (batch,time_step,input_size)
    output = rnn(b_x)                               # rnn output
    loss = loss_func(output, b_y)                   # cross entropy loss
    optimizer.zero_grad()                           # clear gradients for this training step
    loss.backward()                                 # backpropagation, compute gradients
    optimizer.step()                                # apply gradients
  y_predict=[]
  for step,(a_x, a_y) in enumerate(testloader):
    a_x = a_x.view(-1, 1, INPUT_SIZE).float() 
    test_output = rnn(a_x)
    pred_y = torch.max(test_output, 1)[1]
    y_predict=y_predict+pred_y.numpy().tolist()
  precision, recall, fscore, support = score(y_test, y_predict)
  precision=precision*100
  recall=recall*100
  print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
  for i in range(8):
    print('Label: ', i,'| precision: %.2f' % precision[i],'%','| recall: %.2f' % recall[i],'%','| F1: %.2f' % fscore[i],'| support:', support[i])

import pandas as pd  
import numpy as np  
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split 

class CustomDataset(Dataset):
  def __init__(self, features, labels):
    self.features = features
    self.labels = labels
  def __len__(self):
    return(len(self.labels))
  def __getitem__(self, index):
    feature = self.features[index]
    label = self.labels[index]
    return(feature, label)

def preprocess(dataroot, batch_size):
  df=pd.read_csv(dataroot,header=0,low_memory=False)
  pd.set_option('mode.use_inf_as_na', True) 
  df['Flow Bytes/s']=df['Flow Bytes/s'].astype('float64')
  df[' Flow Packets/s']=df[' Flow Packets/s'].astype('float64')
  df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].mean(),inplace=True)
  df[' Flow Packets/s'].fillna(df[' Flow Packets/s'].mean(),inplace=True)
  df[' Label'] = df[' Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)

  x_train, x_test, y_train,y_test = train_test_split(df.iloc[:,0:-1], df[' Label'], test_size=0.3, random_state=42)
  train_data = CustomDataset(x_train, y_train)
  test_data = CustomDataset(x_test, y_test)
  trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
  return trainloader, testloader

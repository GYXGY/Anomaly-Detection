import pandas as pd  
from time import time
SaveFile_Name = r"E:\dataset\MachineLearningCVE\CICIDS2017.csv"
LoadFile_Name = r"E:\dataset\copy\CICIDS2017.csv"

df=pd.read_csv(LoadFile_Name,header=0,low_memory=False)
size_mapping = {'BENIGN':0, 'FTP-Patator':1,'SSH-Patator':1,'DoS Hulk':2, 'DoS GoldenEye':2,'DoS slowloris':2,'DoS Slowhttptest':2,'Heartbleed':2,'Web Attack � Brute Force':3,'Web Attack � XSS':3,'Web Attack � Sql Injection':3,'Infiltration':4,'Bot':5,'PortScan':6,'DDoS':7}
df[' Label'] = df[' Label'].map(size_mapping)
pd.set_option('mode.use_inf_as_na', True) 
df['Flow Bytes/s']=df['Flow Bytes/s'].astype('float64')
df[' Flow Packets/s']=df[' Flow Packets/s'].astype('float64')
df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].mean(),inplace=True)
df[' Flow Packets/s'].fillna(df[' Flow Packets/s'].mean(),inplace=True)

df.to_csv(SaveFile_Name,index=False,mode='a+')
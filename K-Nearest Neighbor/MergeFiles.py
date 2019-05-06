import pandas as pd  
import os  
from time import time
SaveFile_Name = r"E:\dataset\MachineLearningCVE\CICIDS2017.csv"
LoadFile_Name = "E:\dataset\MachineLearningCVE"

#load csvfiles
file_list = os.listdir(LoadFile_Name)  

t0 = time()
#merge first csvfile
df = pd.read_csv(os.path.join(LoadFile_Name,file_list[0]),low_memory=False) 
df.to_csv(SaveFile_Name,index=False,mode='a+')  
  
#merge the other csvfiles
for i in range(1,len(file_list)):  
    df = pd.read_csv(os.path.join(LoadFile_Name,file_list[i]),low_memory=False)
    df.to_csv(SaveFile_Name,index=False,header=False,mode='a+')
tt = time()-t0
print ("Merge files in {} seconds".format(round(tt,3)))

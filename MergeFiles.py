import pandas as pd  
import os  

SaveFile_Name = r"E:\dataset\MachineLearningCVE\CICIDS2017.csv"
LoadFile_Name = "E:\dataset\MachineLearningCVE"

#load csvfiles
file_list = os.listdir(LoadFile_Name)  

#merge first csvfile

data = pd.read_csv(os.path.join(LoadFile_Name,file_list[0]),low_memory=False) 
data.to_csv(SaveFile_Name,index=False,mode='a+')  
  
#merge the other csvfiles
for i in range(1,len(file_list)):  
    data = pd.read_csv(os.path.join(LoadFile_Name,file_list[i]),low_memory=False)
    data.to_csv(SaveFile_Name,index=False,header=False,mode='a+')
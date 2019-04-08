import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
dataroot="E:\dataset\MachineLearningCVE\CICIDS2017.csv"
TrainData_Name = "E:\dataset\MachineLearningCVE\TrainData.csv"
TestData_Name = "E:\dataset\MachineLearningCVE\TestData.csv"

df=pd.read_csv(dataroot,header=0,low_memory=False)

#train_test_split
t0 = time()
X_train, X_test, y_train,y_test = train_test_split(df.iloc[:,0:-1], df[' Label'], test_size=0.3, random_state=42)
tt = time()-t0
print ("Split dataset in {} seconds".format(round(tt,3)))

#Save csvfiles
TrainData= pd.concat([X_train, y_train], axis=1)
TestData= pd.concat([X_test, y_test], axis=1)
TrainData.to_csv(TrainData_Name,index=False,mode='a+') 
TestData.to_csv(TestData_Name,index=False,mode='a+') 

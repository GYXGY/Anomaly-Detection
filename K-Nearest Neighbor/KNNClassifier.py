import pandas as pd
from time import time
print("Let's start")

#Data preprocessing
#dataroot=r"E:\dataset\MachineLearningCVE\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
dataroot="~/home/szp/guyi/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df=pd.read_csv(dataroot,header=0,low_memory=False)
pd.set_option('mode.use_inf_as_na', True) # convert inf to nan
df['Flow Bytes/s']=df['Flow Bytes/s'].astype('float64')
df[' Flow Packets/s']=df[' Flow Packets/s'].astype('float64')
df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].mean(),inplace=True)
df[' Flow Packets/s'].fillna(df[' Flow Packets/s'].mean(),inplace=True)
df[' Label'] = df[' Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)

#train_test_split
from sklearn.model_selection import train_test_split
t0 = time()
X_train, X_test, y_train,y_test = train_test_split(df.iloc[:,0:-1], df[' Label'], test_size=0.3, random_state=42)
tt = time()-t0
print ("Split dataset in {} seconds".format(round(tt,3)))

#Feature scaling
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
    
#Training KNN classifier
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors = 5, algorithm = 'ball_tree', leaf_size=500)
t0 = time()
classifier.fit(X_train,y_train)
tt = time()-t0
print ("Classifier trained in {} seconds".format(round(tt,3)))

#Testing KNN model
t0 = time()
y_predict = classifier.predict(X_test)
tt = time() - t0
print ("Predicted in {} seconds".format(round(tt,3)))

#Showing Results
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
accuracy = accuracy_score(y_test, y_predict)
print ("Accuracy is {}.".format(round(accuracy,4)))
precision = precision_score(y_test, y_predict)
print ("Precision is {}.".format(round(precision,4)))
recall = recall_score(y_test, y_predict)
print ("Recall is {}.".format(round(recall,4)))
print ("F is {}.".format(round(f1_score(y_test, y_predict),4)))

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
plt.plot(fpr,tpr,marker = 'o')
plt.show()
from sklearn.metrics import auc
AUC = auc(fpr, tpr)
print ("AUC is {}.".format(round(recall,4)))
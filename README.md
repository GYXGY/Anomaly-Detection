# Anomaly-Detection
CICIDS2017 dataset<br>
Pytorch platform<br>

Model 1：KNN<br>
"MergeFiles.py" implements the merging of multiple csv files.<br>
"HoldOut.py" realizes the division of training set and test set by hold out method.<br>
"KNNClassifier.py" establishes K-Nearest Neighbor(KNN) model to achieve anomaly detection.<br>

Model 2：RNN<br>
"MergeFiles.py" implements the merging of multiple csv files.<br>
"Dataset_to_Dataloader.py"turns a csv file into a Pytorch dataloader used for network training and testing.<br>
"LSTM.py"establishes Long Short-Term Memory(LSTM) model to achieve anomaly detection.<br>

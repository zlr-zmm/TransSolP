import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, precision_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from model import *
import matplotlib.gridspec as gridspec
import seaborn as sns

# 1. 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features1,features2,features3, labels):
        self.features1 = torch.tensor(features1.values.astype(np.float32))
        self.features2 = torch.tensor(features2.values.astype(np.float32))
        self.features3 = torch.tensor(features3.values.astype(np.float32))
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.features1+ self.features1 +self.features1)

    def __getitem__(self, index):
        x1 = self.features1[index]
        x2 = self.features2[index]
        x3 = self.features3[index]
        y = self.labels[index]
        return x1, x2, x3, y

class MyDataset(Dataset):
    def __init__(self, features1, features2, features3, labels):
        self.features1 = features1
        self.features2 = features2
        self.features3 = features3
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature1 = self.features1[index]
        feature2 = self.features2[index]
        feature3 = self.features3[index]
        label = self.labels[index]
        return feature1, feature2, feature3, label
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import preprocessing

# 5. 数据读取和预处理
train1 = pd.read_csv("Protrain/dataset/Prics_5fold_ProtT5.csv")
train2 = pd.read_csv("Protrain/dataset/Prics_5foldESM.csv")
train3 = pd.read_csv("Protrain/dataset/Prics_5foldESM1b.csv")
test1 = pd.read_csv("Protrain/dataset/Price_Independent_test_set_prot5.csv")
test2 = pd.read_csv("Protrain/dataset/Price_Independent_test_setESM.csv")
test3 = pd.read_csv("Protrain/dataset/Price_Independent_test_setESM1b.csv")
x_train1 = train1.iloc[:, 1:]
x_train2 = train2.iloc[:, 1:]
x_train3 = train3.iloc[:, 1:]
x_test1 = test1.iloc[:, 1:]
x_test2 = test2.iloc[:, 1:]
x_test3 = test3.iloc[:, 1:]
train_label = train1.iloc[:, 0]
test_label = test1.iloc[:, 0]
# 6. 定义超参数
input_dim1 = 1024
input_dim2 = 1280
num_classes = 2
num_epochs = 10
batch_size = 32
num_folds = 5
# 7. 定义交叉验证
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

best_cnn_mcc = 0
best_transformer_mcc = 0
for model in ['Transformer_CNN_BiLSTM']:

    print(model)
    print("Test Accuracy | Precision | MCC | AUC-ROC")
    best_model = []
    best_acc = 0
    for fold, (train_index, val_index) in enumerate(kf.split(x_train1)):
        # print(f"Fold: {fold+1}")
        train_data1 = x_train1.iloc[train_index]
        train_data2 = x_train2.iloc[train_index]
        train_data3 = x_train3.iloc[train_index]
        train_labels = train_label.iloc[train_index]
        val_data1 = x_train1.iloc[val_index]
        val_data2 = x_train2.iloc[val_index]
        val_data3 = x_train3.iloc[val_index]
        val_labels = train_label.iloc[val_index]
        # 9. 创建训练和验证集的数据加载器
        train_dataset = CustomDataset(train_data1, train_data2, train_data3, train_labels)
        val_dataset = CustomDataset(val_data1, val_data2, val_data3, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        test_data1 = torch.tensor(x_test1.values, dtype=torch.float32)
        test_data2 = torch.tensor(x_test2.values, dtype=torch.float32)
        test_data3 = torch.tensor(x_test3.values, dtype=torch.float32)
        test_labels = torch.tensor(test_label.values, dtype=torch.long)
        test_dataset = MyDataset(test_data1, test_data2, test_data3, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        # 10. 创建并训练CNN模型
        if (model == 'CNN_GRU'):
            cnn_model = CNN_GRU(input_dim1,input_dim2,  num_classes)
        elif(model=='CNN_BiLSTM'):
            cnn_model = CNN_BiLSTM(input_dim1,input_dim2,  num_classes)
        elif(model=='GRU_BiLSTM'):
            cnn_model = BiLSTM_GRU(input_dim1,input_dim2,  num_classes)
        elif(model=='Transformer_CNN_GRU'):
            cnn_model = CNN_Transformer_GRU(input_dim1,input_dim2,  num_classes)
        elif (model=='Transformer_GRU_BiLSTM'):
            cnn_model = Transformer_BiLSTM_GRU(input_dim1,input_dim2,  num_classes)
        elif (model=='CNN'):
            cnn_model = CNN(input_dim1,input_dim2,  num_classes)
        elif (model=='Transformer'):
            cnn_model = Transformer(input_dim1,input_dim2,  num_classes)
        elif (model=='GRU'):
            cnn_model = GRU(input_dim1,input_dim2,  num_classes)
        elif (model=='BiLSTM'):
            cnn_model = BiLSTM(input_dim1,input_dim2,  num_classes)
        elif (model=='Transformer_CNN'):
            cnn_model = CNN_Transformer(input_dim1,input_dim2,  num_classes)
        elif (model=='Transformer_GRU'):
            cnn_model = Transformer_GRU(input_dim1,input_dim2,  num_classes)
        elif (model=='Transformer_BiLSTM'):
            cnn_model = Transformer_BiLSTM(input_dim1,input_dim2,  num_classes)
        elif (model=='Transformer_CNN_BiLSTM'):
            cnn_model = CNN_Transformer_BiLSTM(input_dim1,input_dim2,  num_classes)
        elif (model=='CNN_GRU_BiLSTM'):
            cnn_model = CNN_BiLSTM_GRU(input_dim1,input_dim2,  num_classes)
        elif (model=='CNN_GRU_BiLSTM_Transformer'):
            cnn_model = CNN_Transformer_BiLSTM_ems1b_esm(input_dim1,input_dim2,  num_classes)
        elif (model=='Compare'):
            cnn_model = Comp_CNN(input_dim1,input_dim2,  num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

        cnn_model.train()
        for epoch in range(num_epochs):
            all_predictions = []
            all_labels = []
            for data1, data2, data3, labels in train_loader:
                    optimizer.zero_grad()
                    x1_cnn, x1_trans, x1_bilstm,x2_cnn, x2_trans, x2_bilstm,x3_cnn, x3_trans, x3_bilstm,outputs = cnn_model(data1.unsqueeze(1),data3.unsqueeze(1),data2.unsqueeze(1))
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    weights = [x1_cnn.squeeze().detach().numpy(),
                               x2_cnn.squeeze().detach().numpy(),
                               x3_cnn.squeeze().detach().numpy(),
                               x1_trans.squeeze().detach().numpy(),
                               x2_trans.squeeze().detach().numpy(),
                               x3_trans.squeeze().detach().numpy(),
                               x1_bilstm.squeeze().detach().numpy(),
                               x2_bilstm.squeeze().detach().numpy(),
                               x3_bilstm.squeeze().detach().numpy()]
                    weightsname = ["x1_cnn", "x2_cnn","x3_cnn",
                               "x1_trans","x2_trans","x3_trans",
                               "x1_bilstm", "x2_bilstm","x3_bilstm"]
                    fig, axes = plt.subplots(3, 3, figsize=(20, 6))

                    for i, ax in enumerate(axes.flat):
                        normalized_weights_2d = np.expand_dims(weights[i], axis=0)
                        sns.heatmap(normalized_weights_2d, cmap='viridis', ax=ax, cbar=True)

                        # Set axis labels and title
                        ax.set_xlabel('Position')
                        ax.set_ylabel('Sample')
                        ax.set_title(weightsname[i])

                    plt.tight_layout()
                    plt.show()

        test_loss = 0.0
        test_correct = 0
        total_samples = 0
        with torch.no_grad():
            all_predictions = []
            all_labels = []

            for data1, data2, data3, labels in test_loader:
                cnn_outputs= cnn_model(data1.unsqueeze(1),data3.unsqueeze(1),data2.unsqueeze(1))
                draw_T_SNE( cnn_outputs, i=0)
                scores = cnn_outputs[:, 1].tolist()
                all_auc.extend(scores)
                _, predicted = torch.max(cnn_outputs.data, 1)
                all_predictions.extend(predicted.tolist())
                all_labels.extend(labels.tolist())
            val_accuracy = accuracy_score(all_labels, all_predictions)
            if val_accuracy > best_acc:
                best_model = cnn_model

    with torch.no_grad():
        all_predictions = []
        all_labels = []
        all_auc = []

        for data1, data2, data3, labels in test_loader:
            x1_cnn, x1_trans, x1_bilstm,x2_cnn, x2_trans, x2_bilstm,x3_cnn, x3_trans, x3_bilstm,cnn_outputs = best_model(data1.unsqueeze(1), data2.unsqueeze(1), data3.unsqueeze(1))
            scores = cnn_outputs[:, 1].tolist()
            all_auc.extend(scores)
            _, predicted = torch.max(cnn_outputs.data, 1)
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
    test_accuracy = accuracy_score(all_labels, all_predictions)
    test_mcc = matthews_corrcoef(all_labels, all_predictions)
    test_sn = recall_score(all_labels, all_predictions)
    test_sp = recall_score(all_labels, all_predictions, pos_label=0)
    test_precision = precision_score(all_labels, all_predictions)
    test_auc_roc = roc_auc_score(all_labels, all_predictions)
    fpr, tpr, _ = roc_curve(all_labels, all_auc)
    precision, recall, _ = precision_recall_curve(all_predictions, all_auc, pos_label=1)

    curve_data1 = pd.DataFrame({'FPR': precision, 'TPR': recall})
    path1 = 'compare/deaw/PR/Ours{}.csv'.format(fold)
    # Save the curve data to a CSV file
    curve_data1.to_csv(path1, index=False)
    curve_data2 = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    path2 = 'compare/deaw/ROC/Ours{}.csv'.format(fold)
    # Save the curve data to a CSV file
    curve_data2.to_csv(path2, index=False)
    print(f"{test_accuracy:.4f} {test_precision:.4f} {test_mcc:.4f} {test_auc_roc:.4f}")    # with torch.no_grad():
    all_predictions = []
    all_labels = []



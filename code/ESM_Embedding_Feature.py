import torch
import os
from bio_embeddings.embed import ESMEmbedder

# CLS averaging processing function
def getCls(vector):
    vector = vector.mean(axis=0)
    return vector

# CLS data generation and writing functions
def data_write(input_data, output_file_name):
    embedder = ESMEmbedder()
    k=0
    for i in input_data:
        print(k)
        print(i[0])
        k = k+1
        embedding = embedder.embed(i[0])
        cls = getCls(embedding)
        # print(cls)
        if not os.path.exists(output_file_name):
            os.system(r"touch {}".format(output_file_name))
        with open(output_file_name, 'a') as f:
            a = []
            for j in cls:
                a.append(float(j))
            f.write(str(a) + " ")
            f.write("\n")

import pandas as pd



path_to_test = "data/dataset/PSI_Biology_solubility_trainset.csv"
path_to_test_cls = "../ensemble/dataset/PSI_Biology_solubility_trainsetESMEmbedder.csv"
dataset_test = []
test_datasets = pd.read_csv(path_to_test).iloc[:,:].values.tolist()
print(len(test_datasets))
data_write(test_datasets, path_to_test_cls)

# **TransSolP: A stack network model using pre-trained protein language models for accurate identification of soluble proteins** 

author: Lingrong Zhanga,  Taigang Liua*

affiliate: College of Information Technology, Shanghai Ocean University, Shanghai 201306, China

Abstract:The solubility of proteins plays a crucial role in various biological phenomena. Insufficient solubility can lead to protein aggregation, impairing protein function and hindering the development of protein-based therapeutics. Therefore, accurate identification of soluble proteins is of utmost importance. Traditional biological identification methods often rely on experienced researchers and consume significant time. Existing computational methods mainly rely on manual feature extraction and machine learning algorithms for classification, but their accuracy still has room for improvement. In this study, we propose TransSolP, a method for soluble protein prediction based on pre-trained protein language and deep learning models. TransSolP leverages three pre-trained protein language models: ESM, ESM1b, and ProtT5, and employs three deep learning methods (Transformer, CNN, BiLSTM) for further model training. By stacking these models, we maximize their advantages. We validate TransSolP on two widely used datasets, the Price dataset and PSI:Biology dataset, and demonstrate superior performance compared to existing state-of-the-art methods. On the independent validation set of the Price dataset, TransSolP achieves an accuracy of 0.882, a 9.4% improvement compared to existing methods, with a precision of 0.889, MCC of 0.63, and AUC of 0.821. The results highlight TransSolP as a powerful and unique computational tool for broad-scale soluble protein identification. Our code and data are available at https://github.com/zlr-zmm/TransSolP.


This project contains the source code for the TransSolP paper, including the code and data used in the experiments. The raw data is stored in the "data" folder, while the code files are stored in the "code" folder.

- "Embedding_Feature.ipynb" is used to generate embedding features for protein sequences using the ProtT5 pre-trained protein language model.
- "ESM_Embedding_Feature.py" is used to generate embedding features for protein sequences using the ESM and ESM1b pre-trained protein language models.
- "model.py" contains the deep learning architecture code for the TransSolP method.
- "train.py" is the code for model training.

How to use?
First, you need to download the pre-trained model for ProtT5 from Hugging Face. You can find it at: https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main. Additionally, you need to install the "bio_embedding" library. Once you have downloaded the required embedding features using the provided code files, you can run "train.py" to train the model.



bio_embeddings=0.2.2

pandas=1.2.0

numpy=1.23.1

scikit-learn=1.0.2

pytorch=2.0.0

python=3.8



If you find this project helpful, we would greatly appreciate it if you could show your support by giving us a star. Thank you in advance for your support!


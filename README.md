# **TransSolP: A hybrid network model using a pre-trained protein language model for accurate identification of soluble proteins** 

author: Lingrong Zhanga,  Taigang Liua*

affiliate: College of Information Technology, Shanghai Ocean University, Shanghai 201306, China

Abstract:





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


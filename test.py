import pandas as pd
import numpy as np
import torch
from torch import nn
from Utils import *
from preprocess import *
from models import ANN, CNN1D, BILSTM
from torch.utils.data import Dataset, DataLoader
import gensim
import argparse

INPUT_SIZE = 100
HIDDEN_STATE = 64
NUM_LAYERS = 4
NUM_CLASSES = 1 #* binary classification
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




path_pytorch_ANN = r'model_files\ANN_best_model.pth'
path_pytorch_CNN1D = r'model_files\CNN1D_best_model.pth'
path_pytorch_LSTM = r'model_files\BILSTM_best_model.pth'



if __name__ == "__main__":
    local_path = r"C:\\Users\\DELL\\OneDrive\\Desktop\\AI projects\\Fake-News-Detection\\word2vec_model\\word2vec.model"
    colab_path = r'/content/drive/MyDrive/Neuromatch/ANN/temp/word2vec_model/word2vec.model'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ANN")
    parser.add_argument("--colab", type=bool, default=False)
    args = parser.parse_args()
    
    if args.colab:
        word2vec_model = gensim.models.Word2Vec.load(colab_path)
        path_pytorch_ANN = r'/content/Fake-News-Detection/model_files/ANN_best_model.pth'
        path_pytorch_CNN1D = r'/content/Fake-News-Detection/model_files/CNN1D_best_model.pth'
        path_pytorch_LSTM = r'/content/Fake-News-Detection/model_files/BILSTM_best_model.pth'
        
        
    else:
        word2vec_model = gensim.models.Word2Vec.load(local_path)
    if args.model == "ANN":
        model = ANN()
        model.load_state_dict(torch.load(path_pytorch_ANN))
        
    elif args.model == "CNN1D":
        model = CNN1D()
        model.load_state_dict(torch.load(path_pytorch_CNN1D))
        
    elif args.model == "LSTM" or args.model == "BILSTM":
        model = BILSTM(INPUT_SIZE, HIDDEN_STATE, NUM_LAYERS, NUM_CLASSES, bidirection=True)
        model.load_state_dict(torch.load(path_pytorch_LSTM))
    else:
        raise Exception("Invalid model name, please choose one of the following: ANN, CNN1D, LSTM, BILSTM")
    
    

    print(f"{model._get_name()} model loaded successfully\n")
    sample = input("Enter a news: ")
    sample = str(sample)
    model.to(DEVICE)
    model.eval()
    print(predict(sample, word2vec_model, model, DEVICE, prepare_for=model._get_name()))

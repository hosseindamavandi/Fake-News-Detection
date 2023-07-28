import pandas as pd
import numpy as np
import torch
from torch import nn
from Utils import *
from preprocess import *
from models import ANN, CNN1D, BILSTM
from torch.utils.data import Dataset, DataLoader

INPUT_SIZE = 100
HIDDEN_STATE = 64
NUM_LAYERS = 4
NUM_CLASSES = 1 #* binary classification
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


path_word2vec = "C:\\Users\\DELL\\OneDrive\\Desktop\\AI projects\\Fake-News-Detection\\word2vec_model\\word2vec.model"
path_pytorch_ANN = r'model_files\ANN_best_model.pth'
path_pytorch_CNN1D = r'model_files\CNN1D_best_model.pth'
path_pytorch_LSTM = r'model_files\BILSTM_best_model.pth'


word2vec_model = Load_word2vec(path_word2vec)





# # sample = preprocess_text(sample)
# # sample = vectorize_text(sample, word2vec_model)
# # sample = torch.tensor(sample).float().to(DEVICE).unsqueeze(0).unsqueeze(1)
# # print(sample.shape)
# # model.eval()
# # model(sample)
# # print(predict(sample, word2vec_model, loaded_ann, DEVICE))


import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ANN")
    args = parser.parse_args()
    
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

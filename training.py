import pandas as pd
import numpy as np
import torch
from torch import nn
from Utils import split_data, TextDataset, Training
from models import ANN, CNN1D, BILSTM
from torch.utils.data import Dataset, DataLoader


loaded_data = np.load(r"cleaned-dataset\data_npy.npy")
X_train, X_test, y_train, y_test = split_data(loaded_data)

training_dataset = TextDataset(X_train, y_train)
testing_dataset = TextDataset(X_test, y_test)
train_loader = DataLoader(dataset=training_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=training_dataset, batch_size=128, shuffle=True)


# * GLOABL VARIABLES
EPOCHS = 150
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
# model = LSTMModel().to(DEVICE)

# * LSTM
INPUT_SIZE = train_loader.dataset[0][0].shape[0]
HIDDEN_STATE = 64
NUM_LAYERS = 4
NUM_CLASSES = 1  # * binary classification

model = BILSTM(INPUT_SIZE, HIDDEN_STATE, NUM_LAYERS, NUM_CLASSES, bidirection=True).to(
    DEVICE
)


loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


if __name__ == "__main__":
    history = Training(
        model,
        train_loader,
        test_loader,
        EPOCHS,
        DEVICE,
        loss,
        optimizer,
        print_every=5,
    )

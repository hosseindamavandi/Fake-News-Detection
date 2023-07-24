from torch import nn
import torch


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 1000)
        self.linear2 = nn.Linear(1000, 500)
        self.linear3 = nn.Linear(500, 100)
        self.linear4 = nn.Linear(100, 20)
        self.linear5 = nn.Linear(20, 1)

        self.dropout1 = nn.Dropout(0.5)
        self.Hl_actication = nn.LeakyReLU()
        self.output_actication = nn.Sigmoid()

    # * L1 Regularization
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    # * L2 Regularization
    def compute_l2_loss(self, w):
        return torch.pow(w, 2).sum()

    def forward(self, x):
        out = self.dropout1(self.Hl_actication(self.linear1(x)))
        out = self.dropout1(self.Hl_actication(self.linear2(out)))
        out = self.dropout1(self.Hl_actication(self.linear3(out)))
        out = self.dropout1(self.Hl_actication(self.linear4(out)))
        out = self.Hl_actication(self.linear5(out))
        out = self.output_actication(out)

        return out

class Linear_block(nn.Module):
    def __init__(self, in_channel, out_channel, activation = nn.LeakyReLU(), dorp_ratio=0.5):
        super().__init__()
        self.Linear_block = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            activation,
            nn.Dropout(dorp_ratio),
        )

    def forward(self, x):
        out = self.Linear_block(x)
        return out

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=64, kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=1, stride=1)
        
        self.mlp = nn.Sequential(
            Linear_block(112, 1),
            # Linear_block(100, 50),
            # Linear_block(50, 20),
            # Linear_block(20, 1),
        )
            
        
        self.dropout1d = nn.Dropout(0.25)
        self.cnn_activation = nn.LeakyReLU()
        self.output_actication = nn.Sigmoid()
        
        
    # * L1 Regularization
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    # * L2 Regularization
    def compute_l2_loss(self, w):
        return torch.pow(w, 2).sum()

    def forward(self, x):
        x = x.unsqueeze(-1)
        out = self.cnn_activation(self.conv1(x))
        out = self.pool(out)
        
        out = self.cnn_activation(self.conv2(out))
        out = self.pool(out)
        
        out = self.cnn_activation(self.conv3(out))
        out = self.pool(out)
        
        out = nn.Flatten()(out)
        #* change the dimension to fit the MLP
        # out = out.squeeze(-1)
        out = self.mlp(out)
        out = self.output_actication(out)

        return out
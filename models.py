from torch import nn
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


    def forward(self,x):
        out = self.dropout1(self.Hl_actication(self.linear1(x)))
        out = self.dropout1(self.Hl_actication(self.linear2(out)))
        out = self.dropout1(self.Hl_actication(self.linear3(out)))
        out = self.dropout1(self.Hl_actication(self.linear4(out)))
        out = self.Hl_actication(self.linear5(out))
        out = self.output_actication(out)

        return out



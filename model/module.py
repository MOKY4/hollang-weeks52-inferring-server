import torch.nn as nn


class MyModule(nn.Module):
    def __init__(self, in_features=4, out_features=19):
        super(MyModule, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_features, 128), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.fc = nn.Linear(64, out_features)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)

        return out

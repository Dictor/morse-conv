import torch

class MorseCNN(torch.nn.Module):
    def __init__(self) -> None:
        super(MorseCNN, self).__init__()

        # L1
        # input : 1 channel, 64 width, N batch
        # after conv : 24 channel, 64 width, N batch
        # after pool : 24 channel, 32 width, N batch
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 24, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2)
        )

        # L2
        # input : 24 channel, 32 width, N batch
        # after conv : 48 channel, 32 width, N batch
        # after pool : 48 channel, 16 width, N batch
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(24, 48, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2)
        )

        # FC
        # input : 48 channel * 16 height
        # output : 37 (alpha 26 + num 10 + other 1)
        self.fc = torch.nn.Linear(16 * 48, 37)

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
import torch
import torch.nn as nn
class Rblock(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        for i in range(2):
            w = nn.Linear(in_dim,in_dim)
            


class BottleBlock(nn.Module):
    def __init__(self, in_dim, bottle_rate):
        super().__init__()
        branch_dim = int(in_dim/bottle_rate)
        self.br1 = nn.Sequential(
                    nn.Linear(in_dim, branch_dim),
                    nn.LeakyReLU(),
                    nn.Linear(branch_dim, branch_dim),
                    nn.LeakyReLU()
        )
        self.br2 = nn.Sequential(
                    nn.Linear(in_dim, branch_dim),
                    nn.LeakyReLU(),
                    nn.Linear(branch_dim, branch_dim),
                    nn.LeakyReLU()
        )
        self.br3 = nn.Sequential(
                    nn.Linear(in_dim, branch_dim),
                    nn.LeakyReLU(),
                    nn.Linear(branch_dim, branch_dim),
                    nn.LeakyReLU()
        )
        self.br4 = nn.Sequential(
                    nn.Linear(in_dim, branch_dim),
                    nn.LeakyReLU(),
                    nn.Linear(branch_dim, branch_dim),
                    nn.LeakyReLU()
        )
        self.concat_fc = nn.Linear(branch_dim * 4, in_dim)
    def forward(self, x):
        out1 = self.br1(x)
        out2 = self.br2(x)
        out3 = self.br3(x)
        out4 = self.br4(x)
        concat_out = torch.cat((out1, out2, out3, out4), dim=1)
        x_out = self.concat_fc(concat_out)
        return x + x_out
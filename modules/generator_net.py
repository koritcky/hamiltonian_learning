import torch
from torch import nn
import numpy as np
from modules.generator import Generator
from modules.main import noise

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network that return Jxx, Jyy, Jzz, h
    """

    def __init__(self, N_spins):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = N_spins * 4 - 3

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)

        # # x are couplings. But we want to return corresponding probabilities, namely diagonal of rhos in e^-(beta*H)
        # return torch.tensor([np.diag(Generator(couplings.numpy()).rho).real for couplings in x.detach()])
        return x



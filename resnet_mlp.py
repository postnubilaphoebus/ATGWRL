import numpy as np
import torch
from torch import nn
import utils.conf as config

class ResNetMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embedding_dimension = config.word_embedding
        self.hidden = config.mlp_hidden
        self.embedding = nn.Linear(self.embedding_dimension, self.hidden)
        self.lin1 = nn.Linear(self.hidden, self.hidden)
        self.lin2 = nn.Linear(self.hidden * 2, self.hidden)
        self.lin3 = nn.Linear(self.hidden, self.hidden)
        self.lin4 = nn.Linear(self.hidden * 2, self.hidden)
        self.lin5 = nn.Linear(self.hidden, self.hidden)
        self.lin6 = nn.Linear(self.hidden * 2, self.hidden)
        self.lin7 = nn.Linear(self.hidden, self.hidden)
        self.lin8 = nn.Linear(self.hidden * 2, self.hidden)
        self.lin9 = nn.Linear(self.hidden, self.hidden)
        self.lin10 = nn.Linear(self.hidden * 2, self.hidden)

    def forward(self, x):
        x = self.embedding(x)
        x = nn.functional.batch_norm(x)
        x = nn.ReLU(x)
        x_1 = self.lin1(x)
        x_1 = nn.functional.batch_norm(x_1)
        x_1 = nn.ReLU(torch.cat((x_1, x)))
        x_2 = self.lin2(x_1)
        x_2 = nn.functional.batch_norm(x_2)
        x_2 = nn.ReLU(x_2)
        x_3 = self.lin3(x_2)
        x_3 = nn.functional.batch_norm(x_3)
        x_3 = nn.ReLU(torch.cat((x_3, x_2)))
        x_4 = self.lin4(x_3)
        x_4 = nn.functional.batch_norm(x_4)
        x_4 = nn.ReLU(x_4)
        x_5 = self.lin_5(x_4)
        x_5 = nn.functional.batch_norm(x_5)
        x_5 = nn.ReLU((torch.cat((x_5, x_4))))
        x_6 = self.lin6(x_5)
        x_6 = nn.functional.batch_norm(x_6)
        x_6 = nn.ReLU(x_6)
        x_7 = self.lin7(x_6)
        x_7 = nn.functional.batch_norm(x_7)
        x_7 = nn.ReLU(torch.cat((x_7, x_6)))
        x_8 = self.lin8(x_7)
        x_8 = nn.functional.batch_norm(x_8)
        x_8 = nn.ReLU(x_8)
        x_9 = self.lin9(x_8)
        x_9 = nn.functional.batch_norm(x_9)
        x_9 = nn.ReLU(torch.cat((x_8, x_9)))
        x_10 = self.lin10(x_9)

        return x_10
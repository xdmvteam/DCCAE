import torch
import torch.nn as nn
import numpy as np
from objectives import cca_loss, dccae_loss


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):  # layer_sizes1 = [1024, 1024, 1024, outdim_size], input_shape1 = 784
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes  # [784,1024,1024,1024,50]
        for l_id in range(len(layer_sizes) - 1):  # len(layer_sizes） = 5  ->  range(4) -> l_id = 0 1 2 3
            if l_id == len(layer_sizes) - 2:      # l_id == 3
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2).double()

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2


class DCCAE(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DCCAE, self).__init__()
        self.encoder1 = MlpNet(layer_sizes1, input_size1).double()
        self.encoder2 = MlpNet(layer_sizes2, input_size2).double()

        self.decoder1 = MlpNet(layer_sizes1[-2::-1] + [input_size1], layer_sizes1[-1]).double()
        self.decoder2 = MlpNet(layer_sizes2[-2::-1] + [input_size2], layer_sizes2[-1]).double()


        self.loss = dccae_loss(outdim_size, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.encoder1(x1)
        output2 = self.encoder2(x2)

        re1 = self.decoder1(output1)
        re2 = self.decoder2(output2)

        return output1, output2, re1, re2
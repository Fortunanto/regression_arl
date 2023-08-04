###############################################################################
# MIT License
#
# Copyright (c) 2020 Jardenna Mohazzab, Luc Weytingh, 
#                    Casper Wortmann, Barbara Brocades Zaalberg
#
# This file contains an implementation of the ARL model prented in "Fairness 
# without Demographics through Adversarially Reweighted Learning" by Lahoti 
# et al..
#
# Author: Jardenna Mohazzab, Luc Weytingh, 
#         Casper Wortmann, Barbara Brocades Zaalberg 
# Date Created: 2021-01-01
###############################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import regressor

class LearnerNN(nn.Module):
    def __init__(
        self, device='cpu'
    ):
        """
        Implements the learner DNN.
        Args:
          embedding_size: list of tuples (n_classes, n_features) containing
                           embedding sizes for categorical columns.
          n_num_cols: number of numerical inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer.
          activation_fn: the activation function to use.
        """
        super().__init__()
        self.device = device

        self.model = regressor.Regressor(2,2)
        

    def forward(self, x):
        """
        The forward step for the learner.
        """

        x = self.model(x)
        return x


class AdversaryNN(nn.Module):
    def __init__(self, device='cpu'):
        """
        Implements the adversary DNN.
        Args:
          embedding_size: list of tuples (n_classes, n_features) containing
                          embedding sizes for categorical columns.
          n_num_cols: number of numerical inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer.
        """
        super().__init__()
        self.device = device

        self.model = regressor.Regressor(2,1)
        # We want to fine tune the last layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        The forward step for the adversary.
        """
        x = self.model(x)

        x = self.sigmoid(x)
        x_mean = torch.mean(x)
        x = x / torch.max(torch.Tensor([x_mean, 1e-4]))
        x = x + torch.ones_like(x)

        return x


class ARL(nn.Module):

    def __init__(
        self,
        batch_size=256,
        device='cuda',
    ):
        """
        Combines the Learner and Adversary into a single module.

        Args:
          embedding_size: list of tuples (n_classes, embedding_dim) containing
                    embedding sizes for categorical columns.
          n_num_cols: the amount of numerical columns in the data.
          learner_hidden_units: list of ints, specifies the number of units
                    in each linear layer for the learner.
          adversary_hidden_units: list of ints, specifies the number of units
                    in each linear layer for the learner.
          batch_size: the batch size.
          activation_fn: the activation function to use for the learner.
        """
        super().__init__()
        torch.autograd.set_detect_anomaly(True)

        self.device = device
        self.adversary_weights = torch.ones(batch_size, 1)

        self.learner = LearnerNN(
            device=device
        )
        self.adversary = AdversaryNN(
            device=device
        )

        self.learner.to(device)
        self.adversary.to(device)


    def learner_step(self, x):
        self.learner.zero_grad()
        prediction = self.learner(x)
        return prediction

    def adversary_step(self, x):
        """
        Performs one loop
        """
        self.adversary.zero_grad()

        adversary_weights = self.adversary(x)
        return adversary_weights

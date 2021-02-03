import torch.nn.functional as F
import json
import torch.nn as nn
import numpy as np
import torch


############################################################################
#  Losses
############################################################################


class PointLoss(nn.Module):
    """ Loss function which reconstructs 3D points from blendshapes
        and measures distance between Y and YHAT

    Attributes:
      data (int): json data containing 3d points for face and blendshapes
      bs_list (list): keys for json dictionary
      face (torch.tensor): tensor containing 3d points for face
      mult (float): multiplier for loss
      bs_tensor (torch.tesnor): tensor containing 3d points for blendshapes
      crit (nn.Module): loss function
    """
    def __init__(self, blendshapes='./data/bs_points_a.json'):
        """ Initiate class

          Args:
            blendshapes (string): json file containing blendshapes
          """

        super(PointLoss, self).__init__()

        with open(blendshapes) as json_file:
            self.data = json.load(json_file)

        self.bs_list = ['BS.Mesh'] + [f'BS.Mesh{num}' for num in range(1, 51)]
        self.register_buffer('face', torch.FloatTensor(
            np.array([self.data['default'][k] for k in self.data['default'].keys()])))
        self.mult = .0025

        bs_list = []
        for key in self.bs_list:
            bs_list.append([self.data['blend_shapes'][key][k] for k in self.data['default'].keys()])

        self.register_buffer('bs_tensor', torch.FloatTensor(np.array(bs_list)))
        self.crit = nn.L1Loss()

    def forward(self, y_hat, y):
        """ Forward pass

          Args:
            y_hat (torch.tensor): predicted blendshape values
            y (torch.tensor): actual blendshape values

          Returns:
            torch.tensor: loss
          """

        y_hat_blend_weighted = self.bs_tensor * y_hat[:, :, None, None]

        y_hat_face = self.face + y_hat_blend_weighted.sum(dim=1)

        y_blend_weighted = self.bs_tensor * y[:, :, None, None]
        y_face = self.face + y_blend_weighted.sum(dim=1)

        dist = F.pairwise_distance(torch.transpose(y_hat_face, 1, 2),
                                   torch.transpose(y_face, 1, 2))

        return dist.sum() * self.mult


class LogCoshLoss(torch.nn.Module):
    """ logarithm of the hyperbolic cosine loss function """
    def __init__(self):
        super().__init__()

    def forward(self, y, y_hat):
        """ Forward pass

          Args:
            y_hat (torch.tensor): predicted blendshape values
            y (torch.tensor): actual blendshape values

          Returns:
            torch.tensor: loss
          """

        loss = y - y_hat
        return torch.mean(torch.log(torch.cosh(loss + 1e-12)))*2.0
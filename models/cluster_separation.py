# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterSeparatorModule(nn.Module):
    def __init__(self, E, num_points, max_classes):
        super().__init__() 
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(E,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,max_classes,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, proposals, end_points):
        # proposals.shape (B, E, 1)
        proposals = proposals.repeat(1, 1, self.num_points) #(B, E, N)
        net = F.relu(self.bn1(self.conv1(proposals))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        cluster_separation = self.conv3(net)              # (B, C, N)
        end_points['cluster_separation'] = cluster_separation
        return end_points

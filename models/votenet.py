# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.dump_helper import dump_results
from models.loss_helper import get_loss, VoteNetLoss


class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling)

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], end_points)
                
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        end_points = self.pnet(xyz, features, end_points)

        return end_points

class VoteNetModule(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.model = VoteNet(num_class=self.opt.num_class,
                             num_heading_bin=self.opt.num_head_bin,
                             num_size_cluster=self.opt.num_size_cluster,
                             mean_size_arr=self.opt.mean_size_arr,
                             num_proposal=self.opt.num_proposal,
                             input_feature_dim=self.opt.input_feature_dim,
                             vote_factor=self.opt.vote_factor,
                             sampling=self.opt.sampling)
        self.criterion = VoteNetLoss(num_class=self.opt.num_class, num_heading_bin=self.opt.num_head_bin, num_size_cluster=self.opt.num_size_cluster, mean_size_arr=self.opt.mean_size_arr)

    def forward(self, batch, batch_idx, name):
        B = batch["point_cloud"].shape[0]
        end_points = self.model(batch)
        loss = self.criterion(end_points)
        self.log("{}_loss".format(name), loss, prog_bar=True, on_epoch=True, batch_size=B)
        return loss

    def training_step(self, batch, batch_idx):
        return self(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay)
        scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: np.power(self.opt.learning_rate_decay, self.global_step)),
                'interval': 'step',
                'frequency': 1,
                'strict': True,
            }
        return [optimizer], [scheduler]
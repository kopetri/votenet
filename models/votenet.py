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
from models.loss_helper import VoteLoss, HeadLoss, SizeLoss, ObjectnessLoss, CenterLoss, SematicLoss
from utils.nn_distance import nn_distance


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
        end_points = inputs
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
        #self.criterion = VoteNetLoss(num_class=self.opt.num_class, num_heading_bin=self.opt.num_head_bin, num_size_cluster=self.opt.num_size_cluster, mean_size_arr=self.opt.mean_size_arr)
        self.vote_loss       = VoteLoss()
        self.objectness_loss = ObjectnessLoss()
        self.size_loss       = SizeLoss(self.opt.num_size_cluster, self.opt.mean_size_arr)
        self.head_loss       = HeadLoss(self.opt.num_head_bin)
        self.center_loss     = CenterLoss()
        self.sem_loss        = SematicLoss()

    def forward(self, batch, batch_idx, name):
        B = batch["point_clouds"].shape[0]

        end_points = self.model(batch)

        _, object_assignment, _, _ = nn_distance(batch["aggregated_vote_xyz"], end_points['center_label'][:,:,0:3])
        seed_xyz = end_points['seed_xyz']
        vote_xyz = end_points['vote_xyz']
        seed_inds = end_points['seed_inds']
        vote_label_mask = end_points['vote_label_mask']
        vote_label = end_points['vote_label']
        objectness_scores = end_points['objectness_scores']
        aggregated_vote_xyz = end_points['aggregated_vote_xyz']
        center_label = end_points['center_label']
        size_scores = end_points['size_scores']
        size_class_label = end_points['size_class_label']
        size_residual_label = end_points['size_residual_label']
        size_residuals_normalized = end_points['size_residuals_normalized']
        objectness_label = end_points['objectness_label']
        heading_class_label = end_points['heading_class_label']
        heading_scores = end_points['heading_scores']
        heading_residual_label = end_points['heading_residual_label']
        heading_residuals_normalized = end_points['heading_residuals_normalized']
        box_label_mask = end_points['box_label_mask']
        pred_center = end_points['pred_center']
        sem_cls_label = end_points['sem_cls_label']
        sem_cls_scores = end_points['sem_cls_scores']
        
        vl       = self.vote_loss(seed_xyz, vote_xyz, seed_inds, vote_label_mask, vote_label)
        ol       = self.objectness_loss(objectness_scores, aggregated_vote_xyz, center_label)
        scl, srl = self.size_loss(size_scores, size_class_label, size_residual_label, size_residuals_normalized, object_assignment, objectness_label)
        hcl, hrl = self.head_loss(heading_class_label, heading_scores, heading_residual_label, heading_residuals_normalized, object_assignment, objectness_label)
        cl       = self.center_loss(pred_center, center_label, box_label_mask, objectness_label)
        seml     = self.sem_loss(sem_cls_scores, sem_cls_label, object_assignment, objectness_label)

        loss = self.compute_votenet_loss(
            center_loss=cl,
            objectness_loss=ol,
            heading_cls_loss=hcl,
            heading_reg_loss=hrl,
            size_cls_loss=scl,
            size_reg_loss=srl,
            sem_cls_loss=seml,
            vote_loss=vl
        )
        self.log("{}_loss".format(name),             loss, prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_center_loss".format(name),      cl,   prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_objectness_loss".format(name),  ol,   prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_heading_cls_loss".format(name), hcl,  prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_heading_reg_loss".format(name), hrl,  prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_size_cls_loss".format(name),    scl,  prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_size_reg_loss".format(name),    srl,  prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_sem_cls_loss".format(name),     seml, prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_vote_loss".format(name),        vl,   prog_bar=True, on_epoch=True, batch_size=B)
        return loss

    def compute_votenet_loss(self, vote_loss, objectness_loss, center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss):
        box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss
        loss = vote_loss + 0.5 * objectness_loss + box_loss + 0.1 * sem_cls_loss
        loss *= 10
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

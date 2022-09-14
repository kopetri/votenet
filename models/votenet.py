# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.loss_helper import VoteLoss, HeadLoss, SizeLoss, ObjectnessLoss, CenterLoss, SematicLoss, compute_object_label_mask
from utils.nn_distance import nn_distance
from utils.scatterplot import draw_scatterplot


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

        # Proposal Filtering
        self.prob_filter = ProposalFilter(C=2+3+num_heading_bin*2+num_size_cluster*4+num_class)

        # Point to cluster segmentation
        self.point2cluster = PointToClusterModule(n_point_feat=0, C=2+3+num_heading_bin*2+num_size_cluster*4+num_class)

    def compute_point_to_cluster_labels(self, end_points):
        point_features = end_points['points_clouds']# (B, N, P)
        xyz = point_features[:, 0:3]
        pred_centers = end_points['center']
        proposals = end_points['proposals']         # (B, C, K)
        proposal_mask = end_points['proposal_mask'] # (B, K)
        sematic_labels = end_points['semantic_labels'] # (B, N)
        labels = torch.zeros((point_features.shape[0], point_features.shape[1])) # (B, N)
        return end_points

    def compute_proposal_mask(self, end_points):
        pred_centers = end_points['center']
        gt_centers = end_points['center_label']
        # pred_centers.shape (B, K, 3)
        # gt_centers.shape (B, N, 3)
        mask = torch.zeros((pred_centers.shape[0], pred_centers.shape[1])) # (B, K)

        N = pred_centers.shape[1]
        M = gt_centers.shape[1]
        C = pred_centers.shape[-1]
        X = torch.repeat_interleave(pred_centers, M, dim=1)
        Y = gt_centers.repeat(1, N, 1)

        diff = X - Y
        diff = torch.pow(diff, 2)
        diff = torch.sum(diff, dim=-1)
        dist = torch.sqrt(diff)
        dist = dist.reshape(-1, pred_centers.shape[1], gt_centers.shape[1])
        
        y = torch.argmin(dist, dim=1) # (B, 2)
        x = torch.arange(y.shape[0]).unsqueeze(1).repeat(1,y.shape[1]) 
        indices = torch.stack([x,y], dim=2).view(y.shape[1] * y.shape[0], y.shape[1]).permute(1,0).tolist()
        #for ind, m in zip(indices, mask):
        #    for i in ind:
        #        m[i] = 1.0
        mask[indices] = 1.0
        end_points['proposal_mask'] = mask.to(pred_centers)
        return end_points


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

        end_points = self.prob_filter(end_points)
        end_points = self.compute_proposal_mask(end_points)

        end_points = self.compute_point_to_cluster_labels(end_points)
        end_points = self.point2cluster(end_points)

        return end_points

class ProposalFilter(torch.nn.Module):
    def __init__(self, C, size=256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            torch.nn.Linear(C, size),
            torch.nn.Linear(size, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, end_points):
        x = end_points['proposals']
        # x.shape (B, C+3, number_proposals)
        end_points['proposal_pred'] = self.mlp(x.permute(0, 2, 1)).squeeze(-1)
        return end_points

class PointToClusterModule(torch.nn.Module):
    def __init__(self, n_point_feat, C, size=256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            torch.nn.Linear(n_point_feat + C, size),
            torch.nn.Linear(size, 1)
        )

    def forward(self, end_points):
        proposals = end_points["proposals"]
        point_feat = end_points["point_clouds"]
        # point_feat.shape (B, N, P)
        # proposals.shape (B, C, K)
        proposals = proposals.permute(0,2,1) # (B, K, C)
        point_feat.unsqueeze(2).repeat(1, 1, proposals.shape[1], 1)
        proposals = proposals.unsqueeze(1).repeat(1, point_feat.shape[1], 1, 1)
        feat = torch.cat([point_feat, proposals], dim=-1) # (B, N, K, P+C)
        feat = self.mlp(feat).squeeze(-1) # (B, N, K)
        feat = feat.softmax(dim=-1)
        end_points['point_to_cluster_probabilities'] = feat
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
        self.prop_loss       = nn.BCELoss()

    def forward(self, batch, batch_idx, name):
        B = batch["point_clouds"].shape[0]

        end_points = self.model(batch)

        _, object_assignment, _, _ = nn_distance(batch["aggregated_vote_xyz"], end_points['center_label'][:,:,0:3])
        objectness_label, objectness_mask = compute_object_label_mask(batch["aggregated_vote_xyz"], end_points['center_label'])
        
        vl       = self.vote_loss(end_points['seed_xyz'], end_points['vote_xyz'], end_points['seed_inds'], end_points['vote_label_mask'], end_points['vote_label'])
        ol       = self.objectness_loss(end_points['objectness_scores'], objectness_label, objectness_mask)
        scl, srl = self.size_loss(end_points['size_scores'], end_points['size_class_label'], end_points['size_residual_label'], end_points['size_residuals_normalized'], object_assignment, objectness_label)
        hcl, hrl = self.head_loss(end_points['heading_class_label'], end_points['heading_scores'], end_points['heading_residual_label'], end_points['heading_residuals_normalized'], object_assignment, objectness_label)
        cl       = self.center_loss(end_points['center'], end_points['center_label'], end_points['box_label_mask'], objectness_label)
        seml     = self.sem_loss(end_points['sem_cls_scores'], end_points['sem_cls_label'], object_assignment, objectness_label)
        probl    = self.prop_loss(end_points['proposal_pred'], end_points['proposal_mask'])
        
        box_loss = self.compute_box_loss(cl, hcl, hrl, scl, srl)
        loss = self.compute_votenet_loss(vl, ol, box_loss, seml) + probl

        self.log("{}_loss".format(name),             loss,       prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_center_loss".format(name),      cl,         prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_objectness_loss".format(name),  ol,         prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_heading_cls_loss".format(name), hcl,        prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_heading_reg_loss".format(name), hrl,        prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_size_cls_loss".format(name),    scl,        prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_size_reg_loss".format(name),    srl,        prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_sem_cls_loss".format(name),     seml,       prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_vote_loss".format(name),        vl,         prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_box_loss".format(name),         box_loss,   prog_bar=True, on_epoch=True, batch_size=B)
        self.log("{}_prop_loss".format(name),        probl,      prog_bar=True, on_epoch=True, batch_size=B)
        return loss

    def compute_votenet_loss(self, vote_loss, objectness_loss, box_loss, sem_cls_loss):
        loss = vote_loss + 0.5 * objectness_loss + box_loss + 0.1 * sem_cls_loss
        loss *= 10
        return loss

    def compute_box_loss(self, center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss):
        return center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss

    def training_step(self, batch, batch_idx):
        return self(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        end_points = self.model(batch)

        points = batch["point_clouds"].squeeze(0).cpu().numpy() # (N, 3)
        gt_centers = batch['center_label'].squeeze(0).cpu().numpy() # (2, 3)
        pred_centers = end_points['center'].squeeze(0).cpu().numpy()
        dim = batch['bbox_dim'].squeeze(0).cpu().numpy() # (2, 3)

        bbox = np.concatenate([gt_centers, dim], axis=1)
        img = draw_scatterplot(points, pred=pred_centers, bbox=bbox)
        img = img[...,::-1]
        return img, batch["plot_id"].squeeze(0).cpu().item()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay)
        scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: np.power(self.opt.learning_rate_decay, self.global_step)),
                'interval': 'step',
                'frequency': 1,
                'strict': True,
            }
        return [optimizer], [scheduler]

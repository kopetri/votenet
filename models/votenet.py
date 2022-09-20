# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""
import torch
import torch.nn as nn
import numpy as np
from pytorch_utils.module import LightningModule
from models.backbone_module import Pointnet2Backbone
from models.pointnet2 import Pointnet2Backbone as Pointnet2Features
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.loss_helper import VoteLoss, SegmentationLoss, SizeLoss, ObjectnessLoss, CenterLoss, SematicLoss, compute_object_label_mask, compute_segmentation_labels
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

    def __init__(self, input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Point Features
        if self.input_feature_dim > 0:
            self.backbone_feat = Pointnet2Features(output_feature_dim=self.input_feature_dim)
        else:
            self.backbone_feat = None

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_proposal, sampling)


        # Point to cluster segmentation
        self.seg_net = SegmentationModule(n_point_feat=self.input_feature_dim, C=2+3)

    def compute_point_to_cluster_labels(self, end_points):
        point_features = end_points['point_clouds']# (B, N, P)
        xyz = point_features[:,:, 0:3] # (B, N, 3)
        pred_centers = end_points['center'] # (B, K, 3)
        proposal_mask = end_points['proposal_mask'] # (B, K)
        proposal_mask = (~proposal_mask.bool()).float() # invert

        dist = self.compute_distance_A_B(pred_centers, xyz) # (B, K, N)
        proposal_mask = proposal_mask.unsqueeze(-1).repeat(1,1,dist.shape[-1]) * 100 # (B, K, N)
        dist += proposal_mask # make invalid proposal distances very high!
        end_points['point_to_cluster_labels'] = torch.argmin(dist, dim=1) # (B, N)
        return end_points

    def compute_distance_A_B(self, A, B):
        N = A.shape[1]
        M = B.shape[1]
        X = torch.repeat_interleave(A, M, dim=1)
        Y = B.repeat(1, N, 1)

        diff = X - Y
        diff = torch.pow(diff, 2)
        diff = torch.sum(diff, dim=-1)
        dist = torch.sqrt(diff)
        dist = dist.reshape(-1, A.shape[1], B.shape[1])
        return dist

    def compute_proposal_mask(self, end_points):
        pred_centers = end_points['center']
        gt_centers = end_points['center_label']
        # pred_centers.shape (B, K, 3)
        # gt_centers.shape (B, N, 3)
        mask = torch.zeros((pred_centers.shape[0], pred_centers.shape[1])) # (B, K)

        dist = self.compute_distance_A_B(pred_centers, gt_centers)
        
        y = torch.argmin(dist, dim=1) # (B, 2)
        x = torch.arange(y.shape[0]).unsqueeze(1).repeat(1,y.shape[1]).to(y) 
        indices = torch.stack([x,y], dim=2).view(y.shape[1] * y.shape[0], 2).permute(1,0).tolist()
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

        if self.backbone_feat:
            # generate features
            end_points = self.backbone_feat(end_points)

        end_points = self.backbone_net(end_points['point_clouds'], end_points)
                
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

        end_points = self.seg_net(end_points)

        return end_points

class SegmentationModule(torch.nn.Module):
    def __init__(self, n_point_feat, C, K,) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            torch.nn.Linear((3 + n_point_feat + C) * K, 2)
        )

    def forward(self, end_points):
        proposals = end_points["proposals"]
        point_feat = end_points["point_clouds"]
        B = point_feat.shape[0]
        N = point_feat.shape[1]
        # point_feat.shape (B, N, P)
        # proposals.shape (B, C, K)
        proposals = proposals.permute(0,2,1) # (B, K, C)
        point_feat = point_feat.unsqueeze(2).repeat(1, 1, proposals.shape[1], 1)
        proposals = proposals.unsqueeze(1).repeat(1, point_feat.shape[1], 1, 1)
        feat = torch.cat([point_feat, proposals], dim=-1) # (B, N, K, P+C)
        feat = feat.view(B, N, -1) # (B, N, (3+P+C) * K)
        feat = self.mlp(feat).squeeze(-1) # (B, N, 2)
        segmentation_probabilities = feat.permute(0,2,1) # (B, 2, N)
        end_points['segmentation_pred'] = segmentation_probabilities
        return end_points

class VoteNetModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super(VoteNetModule, self).__init__(*args, **kwargs)
        self.model = VoteNet(num_proposal=self.opt.num_proposal,
                             input_feature_dim=self.opt.input_feature_dim,
                             vote_factor=self.opt.vote_factor,
                             sampling=self.opt.sampling)
        self.vote_loss         = VoteLoss()
        self.objectness_loss   = ObjectnessLoss()
        #self.size_loss         = SizeLoss(self.opt.num_size_cluster, self.opt.mean_size_arr)
        #self.head_loss         = HeadLoss(self.opt.num_head_bin)
        self.center_loss       = CenterLoss()
        self.segmentation_loss = SegmentationLoss()
        #self.sem_loss          = SematicLoss()
        #self.prop_loss         = nn.BCELoss()
        #self.segmentation_loss = nn.CrossEntropyLoss(reduction='none') 

    def forward(self, batch, batch_idx, split):
        B = batch["point_clouds"].shape[0]

        end_points = self.model(batch)

        #_, object_assignment, _, _ = nn_distance(batch["aggregated_vote_xyz"], end_points['center_label'][:,:,0:3])
        objectness_label, objectness_mask = compute_object_label_mask(batch["aggregated_vote_xyz"], end_points['center_label'])
        segmentation_label = compute_segmentation_labels(end_points['center'], end_points['center_label'], batch["point_clouds"], batch['noise_label'])
        
        vl       = self.vote_loss(end_points['seed_xyz'], end_points['vote_xyz'], end_points['seed_inds'], end_points['vote_label_mask'], end_points['vote_label'])
        ol       = self.objectness_loss(end_points['objectness_scores'], objectness_label, objectness_mask)
        #scl, srl = self.size_loss(end_points['size_scores'], end_points['size_class_label'], end_points['size_residual_label'], end_points['size_residuals_normalized'], object_assignment, objectness_label)
        #hcl, hrl = self.head_loss(end_points['heading_class_label'], end_points['heading_scores'], end_points['heading_residual_label'], end_points['heading_residuals_normalized'], object_assignment, objectness_label)
        cl       = self.center_loss(end_points['center'], end_points['center_label'], end_points['box_label_mask'], objectness_label)
        #seml     = self.sem_loss(end_points['sem_cls_scores'], end_points['sem_cls_label'], object_assignment, objectness_label)
        #probl    = self.prop_loss(end_points['proposal_pred'], end_points['proposal_mask'])
        #segl     = torch.mean(self.segmentation_loss(end_points['point_to_cluster_probabilities'], end_points['point_to_cluster_labels']))
        #box_loss = self.compute_box_loss(cl, hcl, hrl, scl, srl)
        #loss = self.compute_votenet_loss(vl, ol, box_loss, seml)
        sl       = self.segmentation_loss(end_points['segmentation_pred'], segmentation_label)
        loss = (vl + ol + cl + sl) / 4.0
        loss *= 10

        self.log_value("loss",             loss,     split=split, batch_size=B)
        self.log_value("center_loss",      cl,       split=split, batch_size=B)
        self.log_value("objectness_loss",  ol,       split=split, batch_size=B)
        #self.log_value("heading_cls_loss", hcl,      split=split, batch_size=B)
        #self.log_value("heading_reg_loss", hrl,      split=split, batch_size=B)
        #self.log_value("size_cls_loss",    scl,      split=split, batch_size=B)
        #self.log_value("size_reg_loss",    srl,      split=split, batch_size=B)
        #self.log_value("sem_cls_loss",     seml,     split=split, batch_size=B)
        self.log_value("vote_loss",        vl,       split=split, batch_size=B)
        #self.log_value("box_loss",         box_loss, split=split, batch_size=B)
        #self.log_value("prop_loss",        probl,    split=split, batch_size=B)
        self.log_value("seg_loss",         sl,     split=split, batch_size=B)
        if batch_idx == 0 and split == "valid":
            self.visualize_prediction(batch, end_points, segmentation_label, log=True)
        return loss

    def compute_votenet_loss(self, vote_loss, objectness_loss, box_loss, sem_cls_loss):
        loss = vote_loss + 0.5 * objectness_loss + box_loss + 0.1 * sem_cls_loss
        loss *= 10
        return loss

    def compute_box_loss(self, center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss):
        return center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss

    def predict_step(self, batch, batch_idx):
        end_points = self.model(batch)
        img_gt, img_pred, points, gt_centers, pred_centers = self.visualize_prediction(batch, end_points, log=False)
        img_pred = img_pred[...,::-1]
        img_gt = img_gt[...,::-1]
        return img_gt, img_pred, batch["plot_id"].squeeze(0).cpu().item(), points, gt_centers, pred_centers

    def visualize_prediction(self, batch, end_points, segmentation_label, log=True):
        points = batch["point_clouds"].squeeze(0).cpu().numpy() # (N, 3)
        gt_centers = batch['center_label'].squeeze(0).cpu().numpy() # (2, 3)
        pred_centers = end_points['center'].squeeze(0).cpu().numpy()
        dim = batch['bbox_dim'].squeeze(0).cpu().numpy() # (2, 3)
        segmentation_pred = end_points['segmentation_pred'].squeeze(0).cpu().softmax(dim=0) # (2, N)
        segmentation_pred = torch.argmax(segmentation_pred, dim=0).numpy() # (N)
        segmentation_label = segmentation_label.squeeze(0).cpu().numpy() # (N)
        objectness_score = end_points['objectness_scores'].squeeze(0).cpu().softmax(dim=1).numpy() # (K, 2)
        objectness_score = np.argmax(objectness_score, axis=1) # (K)

        bbox = np.concatenate([gt_centers, dim], axis=1)
        img_pred = draw_scatterplot(points, pred=pred_centers, bbox=bbox, objectness_score=objectness_score, seg_pred=segmentation_pred)
        img_gt   = draw_scatterplot(points, bbox=bbox, seg_gt=segmentation_label)
        if log: self.log_image(key='valid_pred', images=[img_pred])
        if log: self.log_image(key='valid_gt', images=[img_gt])
        return img_gt, img_pred, points, gt_centers, pred_centers

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay)
        scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: np.power(self.opt.learning_rate_decay, self.global_step)),
                'interval': 'step',
                'frequency': 1,
                'strict': True,
            }
        return [optimizer], [scheduler]

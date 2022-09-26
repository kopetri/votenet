# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2.pointnet2_modules import PointnetSAModuleVotes
from pointnet2 import pointnet2_utils

def compute_adjacent_matrix(proposal_feat):
    # proposal_feat.shape (B, K, E)
    A = proposal_feat.unsqueeze(2) # (B, K, 1, E)
    B = proposal_feat.unsqueeze(1) # (B, 1, K, E)
    # dot product
    m = torch.sum(A * B, dim=3) # (B, K, K)
    # normalize to -1 and 1
    length = torch.sqrt(torch.sum(m**2, dim=1)).unsqueeze(2)
    m = m.div(length)
    # normalize to 0 and 1
    m = (m + 1.0) / 2.0 # (B, K, K)
    m = m.unsqueeze(-1).repeat(1,1,1,2) # (B, K, K, 2)
    m[..., -1] = 1.0 - m[..., -1] 
    return m.permute(0, 3, 1, 2) # (B, 2, K, K)

def decode_scores(net, end_points):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    objectness_scores = net_transposed[:,:,0:2]
    end_points['objectness_scores'] = objectness_scores
    
    base_xyz = end_points['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    end_points['center'] = center
    proposal_features = net_transposed[:,:, 5:] # (batch_size, num_proposal, E)
    end_points['adjacent_matrix'] = compute_adjacent_matrix(proposal_features)
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_proposal, sampling, E=64, seed_feat_dim=256):
        super().__init__() 

        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        # Object proposal/detection
        # Objectness scores (2), center residual (3)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,2+3+E,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            print('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+F, num_proposal)
        end_points['proposals'] = net

        end_points = decode_scores(net, end_points)
        return end_points

if __name__=='__main__':
    from sunrgbd.sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    net = ProposalModule(DC.num_class, DC.num_heading_bin,
        DC.num_size_cluster, DC.mean_size_arr,
        128, 'seed_fps').cuda()
    end_points = {'seed_xyz': torch.rand(8,1024,3).cuda()}
    out = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda(), end_points)
    for key in out:
        print(key, out[key].shape)

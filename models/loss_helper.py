# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from utils.nn_distance import nn_distance, huber_loss

# euclid dist1
# max  10.80859088897705
# min  0.004594038240611553
# mean 0.6839807866623421
# std  0.39462738209756815
FAR_THRESHOLD = 0.4
NEAR_THRESHOLD = 0.1
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.1,0.9] # put larger weights on positive objectness
NOISE_CLS_WEIGHTS = [0.9,0.1] # put larger weights on negative samples

def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(end_points):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    end_points['box_loss'] = box_loss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points


class VoteNetLoss(torch.nn.Module):
    def __init__(self, num_heading_bin, num_size_cluster, num_class, mean_size_arr) -> None:
        super().__init__()
        self.config = Namespace()
        self.config.num_heading_bin = num_heading_bin
        self.config.num_size_cluster = num_size_cluster
        self.config.num_class = num_class
        self.config.mean_size_arr = mean_size_arr

    def forward(self, end_points):
        # Vote loss
        vote_loss = compute_vote_loss(end_points)
        # Obj loss
        objectness_loss, end_points['objectness_label'], end_points['objectness_mask'], end_points['object_assignment'] = compute_objectness_loss(end_points)
  
        # Box loss and sem cls loss
        center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(end_points, self.config)
        box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss

        # Final loss function
        loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
        loss *= 10
        return loss

class SematicLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')

    def forward(self, sem_cls_scores, sem_cls_label, object_assignment, objectness_label):
        # 3.4 Semantic cls loss
        sem_cls_label = torch.gather(sem_cls_label, 1, object_assignment) # select (B,K) from (B,K2)
        sem_cls_loss = self.criterion_sem_cls(sem_cls_scores.transpose(2,1), sem_cls_label) # (B,K)
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
        return sem_cls_loss

class SizeLoss(torch.nn.Module):
    def __init__(self, num_size_cluster, mean_size_arr) -> None:
        super().__init__()
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
                
    def forward(self, size_scores, size_class_label, size_residual_label, size_residuals_normalized, object_assignment, objectness_label):
        batch_size = object_assignment.shape[0]
        # Compute size loss
        size_class_label = torch.gather(size_class_label, 1, object_assignment) # select (B,K) from (B,K2)
        criterion_size_class = nn.CrossEntropyLoss(reduction='none')
        size_class_loss = criterion_size_class(size_scores.transpose(2,1), size_class_label) # (B,K)
        size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

        size_residual_label = torch.gather(size_residual_label, 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
        size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], self.num_size_cluster).zero_()
        size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
        size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
        predicted_size_residual_normalized = torch.sum(size_residuals_normalized*size_label_one_hot_tiled, 2) # (B,K,3)

        mean_size_arr_expanded = torch.from_numpy(self.mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
        mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
        size_residual_label_normalized = size_residual_label / (mean_size_label+1e-6) # (B,K,3)
        size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
        size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)
        return size_class_loss, size_residual_normalized_loss

class HeadLoss(torch.nn.Module):
    def __init__(self, num_heading_bin) -> None:
        super().__init__()
        self.num_heading_bin = num_heading_bin

    def forward(self, heading_class_label, heading_scores, heading_residual_label, heading_residuals_normalized, object_assignment, objectness_label):
        batch_size = object_assignment.shape[0]
        objectness_label = objectness_label.float()

        # Compute heading loss
        heading_class_label = torch.gather(heading_class_label, 1, object_assignment) # select (B,K) from (B,K2)
        criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
        heading_class_loss = criterion_heading_class(heading_scores.transpose(2,1), heading_class_label) # (B,K)
        heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

        heading_residual_label = torch.gather(heading_residual_label, 1, object_assignment) # select (B,K) from (B,K2)
        heading_residual_normalized_label = heading_residual_label / (np.pi/self.num_heading_bin)

        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], self.num_heading_bin).zero_()
        heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
        heading_residual_normalized_loss = huber_loss(torch.sum(heading_residuals_normalized*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
        heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)
        return heading_class_loss, heading_residual_normalized_loss

class CenterLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred_center, center_label, box_label_mask, objectness_label):
        # Compute center loss
        gt_center = center_label[:,:,0:3]
        dist1, _, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
        objectness_label = objectness_label.float()
        centroid_reg_loss1 = torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
        centroid_reg_loss2 = torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
        center_loss = centroid_reg_loss1 + centroid_reg_loss2
        return center_loss

class VoteLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, seed_xyz, vote_xyz, seed_inds, vote_label_mask, vote_label):
        # Load ground truth votes and assign them to seed points
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]   # B,num_seed,3
        vote_xyz = vote_xyz            # B,num_seed*vote_factor,3
        seed_inds = seed_inds.long()   # B,num_seed in [0,num_points-1]

        # Get groundtruth votes for the seed points
        # vote_label_mask: Use gather to select B,num_seed from B,num_point
        #   non-object point has no GT vote mask = 0, object point has mask = 1
        # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
        #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
        seed_gt_votes_mask = torch.gather(vote_label_mask, 1, seed_inds)
        seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
        seed_gt_votes = torch.gather(vote_label, 1, seed_inds_expand)
        seed_gt_votes += seed_xyz.repeat(1,1,3)

        # Compute the min of min of distance
        vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
        seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
        # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
        _, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
        votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
        votes_dist = votes_dist.view(batch_size, num_seed)
        vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
        return vote_loss

class ObjectnessLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, objectness_scores, objectness_label, objectness_mask):
        # Compute objectness loss
        criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
        objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
        objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)
        return objectness_loss

class AdjacentLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCELoss(reduction='mean')

    def forward(self, pred, gt):
        return self.criterion(pred, gt)

class SegmentationLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(torch.Tensor(NOISE_CLS_WEIGHTS).cuda(), reduction='none')

    def forward(self, segmentation_pred, segmentation_labels):
        # segmentation_pred.shape (B, N, K)
        # segmentation_labels.shape (B, N)
        segmentation_loss = self.criterion(segmentation_pred, segmentation_labels)
        return torch.mean(segmentation_loss)

def compute_adjacents_labels(pred_centers, gt_centers, objectmask):
    def make_adjacent_matrix(vec):
        # vec.shape (K)
        K = len(vec)
        matrix = torch.zeros((K, K))
        for i in torch.unique(vec):
            combs = torch.combinations(torch.where(vec==i)[0], with_replacement=True).permute(1,0)
            combs_inverse = torch.flip(combs, [0])

            matrix[combs.tolist()] = 1
            matrix[combs_inverse.tolist()] = 1
        return matrix
    def __compute_distance_A_B(A, B):
        N = A.shape[1]
        M = B.shape[1]
        X = A.unsqueeze(2).repeat(1, 1, M, 1)
        Y = B.unsqueeze(1).repeat(1, N, 1, 1)

        diff = X - Y
        diff = torch.pow(diff, 2)
        diff = torch.sum(diff, dim=-1)
        dist = torch.sqrt(diff)
        
        return dist
    # pred_centers.shape (B, K, 3)
    # gt_centers.shape   (B, P, 3)
    B = pred_centers.shape[0]
    K = pred_centers.shape[1]
    labels = torch.zeros((B, K, K), dtype=int).to(objectmask)
    dist = __compute_distance_A_B(pred_centers, gt_centers) # (B, K, P)
    clusters = torch.argmin(dist, dim=2) # (B, K)
    for bidx, cluster in enumerate(clusters):
        labels[bidx] = make_adjacent_matrix(cluster)
    
    # objectness_mask (B, K)
    K = objectmask.shape[1]
    objectmask_hor = objectmask.unsqueeze(1).repeat(1,K,1).bool()
    objectmask_ver = objectmask.unsqueeze(2).repeat(1,1,K).bool()
    objectmask = ~torch.logical_or(objectmask_hor, objectmask_ver)
    objectmask = objectmask.float()
    labels = labels * objectmask
    return labels.to(pred_centers)

def compute_segmentation_labels(pred_centers, gt_centers, point_features, noise_label):
    """
    def __compute_distance_A_B(A, B):
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
    mask = torch.zeros((pred_centers.shape[0], pred_centers.shape[1])) # (B, K)
    dist = __compute_distance_A_B(pred_centers, gt_centers)
    
    y = torch.argmin(dist, dim=1) # (B, 2)
    x = torch.arange(y.shape[0]).unsqueeze(1).repeat(1,y.shape[1]).to(y) 
    indices = torch.stack([x,y], dim=2).view(y.shape[1] * y.shape[0], 2).permute(1,0).tolist()
    mask[indices] = 1.0
    proposal_mask = mask.to(pred_centers)

    xyz = point_features[:,:, 0:3] # (B, N, 3)
    proposal_mask = (~proposal_mask.bool()).float() # invert

    dist = __compute_distance_A_B(pred_centers, xyz) # (B, K, N)
    proposal_mask = proposal_mask.unsqueeze(-1).repeat(1,1,dist.shape[-1]) * 100 # (B, K, N)
    dist += proposal_mask # make invalid proposal distances very high!
    point_to_cluster_labels = torch.argmin(dist, dim=1) # (B, N)
    point_to_cluster_labels += 1 # move labels => class idx 0 is noise
    point_to_cluster_labels[noise_label==0] = 0
    return point_to_cluster_labels
    """
    return noise_label

def compute_object_label_mask(aggregated_vote_xyz, center_label):
    gt_center = center_label[:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    dist1, _, _, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1
    return objectness_label, objectness_mask

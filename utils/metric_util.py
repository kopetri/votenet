# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Utility functions for metric evaluation.

Author: Or Litany and Charles R. Qi
"""


import torch
import numpy as np
from torchmetrics import JaccardIndex
from itertools import permutations

# Mesh IO
import trimesh

 
# ----------------------------------------
# Precision and Recall
# ----------------------------------------

def multi_scene_precision_recall(labels, pred, iou_thresh, conf_thresh, label_mask, pred_mask=None):
    '''
    Args:
        labels: (B, N, 6)
        pred: (B, M, 6)
        iou_thresh: scalar
        conf_thresh: scalar
        label_mask: (B, N,) with values in 0 or 1 to indicate which GT boxes to consider.
        pred_mask: (B, M,) with values in 0 or 1 to indicate which PRED boxes to consider.
    Returns:
        TP,FP,FN,Precision,Recall
    '''
    # Make sure the masks are not Torch tensor, otherwise the mask==1 returns uint8 array instead
    # of True/False array as in numpy
    assert(not torch.is_tensor(label_mask))
    assert(not torch.is_tensor(pred_mask))
    TP, FP, FN = 0, 0, 0
    if label_mask is None: label_mask = np.ones((labels.shape[0], labels.shape[1]))
    if pred_mask is None: pred_mask = np.ones((pred.shape[0], pred.shape[1]))
    for batch_idx in range(labels.shape[0]):
        TP_i, FP_i, FN_i = single_scene_precision_recall(labels[batch_idx, label_mask[batch_idx,:]==1, :],
                                                         pred[batch_idx, pred_mask[batch_idx,:]==1, :],
                                                         iou_thresh, conf_thresh)
        TP += TP_i
        FP += FP_i
        FN += FN_i
    
    return TP, FP, FN, precision_recall(TP, FP, FN)
      

def single_scene_precision_recall(labels, pred, iou_thresh, conf_thresh):
    """Compute P and R for predicted bounding boxes. Ignores classes!
    Args:
        labels: (N x bbox) ground-truth bounding boxes (6 dims) 
        pred: (M x (bbox + conf)) predicted bboxes with confidence and maybe classification
    Returns:
        TP, FP, FN
    """
    
    
    # for each pred box with high conf (C), compute IoU with all gt boxes. 
    # TP = number of times IoU > th ; FP = C - TP 
    # FN - number of scene objects without good match
    
    gt_bboxes = labels[:, :6]      
    
    num_scene_bboxes = gt_bboxes.shape[0]
    conf = pred[:, 6]    
        
    conf_pred_bbox = pred[np.where(conf > conf_thresh)[0], :6]
    num_conf_pred_bboxes = conf_pred_bbox.shape[0]
    
    # init an array to keep iou between generated and scene bboxes
    iou_arr = np.zeros([num_conf_pred_bboxes, num_scene_bboxes])    
    for g_idx in range(num_conf_pred_bboxes):
        for s_idx in range(num_scene_bboxes):            
            iou_arr[g_idx, s_idx] = calc_iou(conf_pred_bbox[g_idx ,:], gt_bboxes[s_idx, :])
    
    
    good_match_arr = (iou_arr >= iou_thresh)
            
    TP = good_match_arr.any(axis=1).sum()    
    FP = num_conf_pred_bboxes - TP        
    FN = num_scene_bboxes - good_match_arr.any(axis=0).sum()
    
    return TP, FP, FN
    

def precision_recall(TP, FP, FN):
    Prec = 1.0 * TP / (TP + FP) if TP+FP>0 else 0
    Rec = 1.0 * TP / (TP + FN)
    return Prec, Rec
    

def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths        
    Returns:
        iou
    """        
        
    max_a = box_a[0:3] + box_a[3:6]/2
    max_b = box_b[0:3] + box_b[3:6]/2    
    min_max = np.array([max_a, max_b]).min(0)
        
    min_a = box_a[0:3] - box_a[3:6]/2
    min_b = box_b[0:3] - box_b[3:6]/2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0*intersection / union

class AdjacentAccuracy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, gt):
        pred = pred.detach()
        gt = gt.detach()
        mask = gt==1
        pred = torch.argmax(pred, dim=1)
        correct = torch.sum(torch.triu(pred==gt, diagonal=1).int()[mask])
        total   = torch.sum(torch.triu(torch.ones_like(gt), diagonal=1)[mask])
        if total == 0: return 0.0
        return correct / total

class ClusterAccuracy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, gt):
        pred = pred.detach()
        gt = gt.detach()
        #pred.shape (B, K)
        num_classes = torch.max(gt, dim=1)[0]
        num_classes = num_classes.unsqueeze(1).repeat(1, gt.shape[1])
        pred = torch.clip(pred, max=num_classes) # clip predicted num_classes to gt num_classes
        correct = torch.sum((pred == gt).int())
        total = torch.sum(torch.ones_like(gt))
        return correct / total

class IoU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, gt, proposal2cluster):
        pred = pred.detach()
        gt = gt.detach()
        proposal2cluster = proposal2cluster.detach()
        B = proposal2cluster.shape[0]
        proposal2cluster = torch.cat([torch.zeros(B,1).to(proposal2cluster), proposal2cluster+1], dim=1).long() # pad with 0 in the front
        jaccard_index = JaccardIndex(num_classes=(torch.max(proposal2cluster)+1).int(), average=None).to(pred)
        pred = pred.softmax(dim=1)
        pred = torch.argmax(pred, dim=1)
        proposal2cluster = proposal2cluster.permute(1,0) # (B, K) -> (K, B)
        pred = proposal2cluster[pred][...,0]
        gt = proposal2cluster[gt][...,0]
        iou = jaccard_index(pred, gt)
        return iou

class MCLAccuracy(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_indices = torch.arange(1, num_classes)

    def forward(self, pred, gt):
        pred = pred.detach() # (B, P, N)
        gt = gt.detach()
        
        pred = torch.argmax(pred, dim=1) # (B, N)
        cluster_iou = 0.0
        noise_iou = 0.0
        for p in permutations(self.class_indices):
            p = torch.tensor(p)
            p = torch.cat([torch.tensor([0]), p])
            try:
                pred_perm = p[pred].to(pred)
                iou = JaccardIndex(average=None, num_classes=self.num_classes).cuda()(pred_perm, gt)
                noise_iou = iou[0].item()
                cluster_iou = max(cluster_iou, torch.mean(iou[1:]).item())
            except:
                pass
        return noise_iou, cluster_iou


if __name__ == '__main__':
    print('running some tests')
    
    ############
    ## Test IoU 
    ############
    box_a = np.array([0,0,0,1,1,1])
    box_b = np.array([0,0,0,2,2,2])
    expected_iou = 1.0/8
    pred_iou = calc_iou(box_a, box_b)
    assert expected_iou == pred_iou, 'function returned wrong IoU'
    
    box_a = np.array([0,0,0,1,1,1])
    box_b = np.array([10,10,10,2,2,2])
    expected_iou = 0.0
    pred_iou = calc_iou(box_a, box_b)
    assert expected_iou == pred_iou, 'function returned wrong IoU'
    
    print('IoU test -- PASSED')
    
    #########################
    ## Test Precition Recall 
    #########################
    gt_boxes = np.array([[0,0,0,1,1,1],[3, 0, 1, 1, 10, 1]])
    detected_boxes = np.array([[0,0,0,1,1,1, 1.0],[3, 0, 1, 1, 10, 1, 0.9]])
    TP, FP, FN = single_scene_precision_recall(gt_boxes, detected_boxes, 0.5, 0.5)
    assert TP == 2 and FP == 0 and FN == 0
    assert precision_recall(TP, FP, FN) == (1, 1)
    
    detected_boxes = np.array([[0,0,0,1,1,1, 1.0]])
    TP, FP, FN = single_scene_precision_recall(gt_boxes, detected_boxes, 0.5, 0.5)
    assert TP == 1 and FP == 0 and FN == 1
    assert precision_recall(TP, FP, FN) == (1, 0.5)
    
    detected_boxes = np.array([[0,0,0,1,1,1, 1.0], [-1,-1,0,0.1,0.1,1, 1.0]])
    TP, FP, FN = single_scene_precision_recall(gt_boxes, detected_boxes, 0.5, 0.5)
    assert TP == 1 and FP == 1 and FN == 1
    assert precision_recall(TP, FP, FN) == (0.5, 0.5)
    
    # wrong box has low confidence
    detected_boxes = np.array([[0,0,0,1,1,1, 1.0], [-1,-1,0,0.1,0.1,1, 0.1]])
    TP, FP, FN = single_scene_precision_recall(gt_boxes, detected_boxes, 0.5, 0.5)
    assert TP == 1 and FP == 0 and FN == 1
    assert precision_recall(TP, FP, FN) == (1, 0.5)
    
    print('Precition Recall test -- PASSED')
    

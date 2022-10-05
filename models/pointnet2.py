import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_utils.module import LightningModule
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
from utils.scatterplot import draw_scatterplot
from torchmetrics import JaccardIndex as IoU
from utils.metric_util import MCLAccuracy


class Pointnet2Backbone(nn.Module):
    def __init__(self, input_feature_dim=0, output_feature_dim=128):
        super(Pointnet2Backbone, self).__init__()
        self.output_feature_dim = output_feature_dim
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3+input_feature_dim, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, output_feature_dim])
        

    def forward(self, end_points):
        xyz = end_points['point_clouds'].permute(0,2,1)
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)


        end_points['point_features'] = l0_points.permute(0, 2, 1)
        end_points['point_clouds'] = torch.cat([end_points['point_clouds'], end_points["point_features"]], dim=2)
        return end_points

class ClusterSeparation(nn.Module):
    def __init__(self, max_cluster, input_feature_dim=0):
        super().__init__()
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3+input_feature_dim, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, max_cluster+2, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        noise = x[:, 0:2, :] # (B, 2, N)
        mcl   = x[:, 2:, :]  # (B, max_cluster, N)
        return noise, mcl

class MCL(nn.Module):
    def __init__(self, weights=torch.tensor([1.0, 1.0])) -> None:
        super().__init__()
        # Meta Classification Likelihood (MCL)
        self.eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
        self.weights = weights
        
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        self.weights = self.weights.to(prob1)
        P = prob1.mul_(prob2)
        P = P.sum(2)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(self.eps).log_()
        
        # scale loss using weights
        similar_mask = simi==1
        neglogP[similar_mask]  *= self.weights[0]
        neglogP[~similar_mask] *= self.weights[1]
        return neglogP.mean()

class KLDiv(nn.Module):
    # Calculate KL-Divergence
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
        
    def forward(self, predict, target):
       assert predict.ndimension()==3,'Input dimension must be 3'
       target = target.clone().detach()

       # KL(T||I) = \sum T(logT-logI)
       predict += KLDiv.eps
       target += KLDiv.eps
       logI = predict.log()
       logT = target.log()
       TlogTdI = target * (logT - logI)
       kld = TlogTdI.sum(2)
       return kld

class KCL(nn.Module):
    # KLD-based Clustering Loss (KCL)

    def __init__(self, margin=2.0, weights=torch.tensor([1.0, 1.0])):
        super(KCL,self).__init__()
        self.kld = KLDiv()
        self.hingeloss = nn.HingeEmbeddingLoss(margin, reduction='none')
        self.weights = weights

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        self.weights = self.weights.to(prob1)

        kld = self.kld(prob1,prob2)
        output = self.hingeloss(kld,simi)
        # scale loss using weights
        similar_mask = simi==1
        output[similar_mask]  *= self.weights[0]
        output[~similar_mask] *= self.weights[1]
        return output.mean()

class NoiseLoss(nn.Module):
    def __init__(self):
        super(NoiseLoss,self).__init__()
        self.noise_weights = [0.9, 0.1]

    def forward(self, pred, gt):
        cel = torch.nn.CrossEntropyLoss(torch.tensor(self.noise_weights)).to(pred)
        return cel(pred, gt)

class MCLKCL(nn.Module):
    def __init__(self, t=0.5, margin=2.0):
        super(MCLKCL,self).__init__()
        self.mcl = MCL()
        self.kcl = KCL(margin=margin)
        self.t = np.clip(t, 0, 1)

    def forward(self, prob1, prob2, simi):
        return self.t * self.mcl(prob1, prob2, simi) + (1.0-self.t) * self.kcl(prob1, prob2, simi)

def PairEnum(x, mask=None):
    # mask.shape (B, N)
    # x.shape (B, N, C)
    B = x.shape[0]
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 3, 'Input dimension must be 3'
    x1 = x.repeat(1, x.size(1), 1) # (B, N*N, C)
    x2 = x.repeat(1, 1, x.size(1)).view(B,-1, x.size(2)) # (B, N*N, C)
    if mask is not None:
        xmask = mask.view(B, -1).repeat(1, x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(1, -1,x.size(2))
        x2 = x2[xmask].view(1, -1,x.size(2))
    return x1,x2


def Class2Simi(x,mode='hinge',mask=None):
    B = x.shape[0]
    # Convert class label to pairwise similarity
    n=x.shape[1]
    expand1 = x.reshape(B,n,1).repeat(1,1,n)
    expand2 = x.reshape(B,1,n).repeat(1,n,1)
    out = expand1 - expand2    
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    if mode=='cls':
        out[out==-1] = 0 #dissimilar pair: label=0
    if mode=='hinge':
        out = out.float() #hingeloss require float type
    out = out.view(B, -1)
    if mask is not None: # apply mask
        mask = mask.view(B, -1).repeat(1, x.size(1))
        out = out[mask]
        out = out.view(1, -1)
    return out

class ClusterSeparationModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = ClusterSeparation(self.opt.max_clusters)
        if self.opt.loss == "mcl": self.criterion = MCL(weights=torch.tensor([0.9, 0.1]))
        if self.opt.loss == "kcl": self.criterion = KCL(weights=torch.tensor([0.9, 0.1]))
        self.noise_loss = NoiseLoss()
        self.acc = MCLAccuracy(num_classes=self.opt.max_clusters)

    def forward(self, batch, batch_idx, split):
        xyz = batch['point_clouds'] # (B, N, 3)
        B = xyz.shape[0]

        # make prediction
        noise_logits, mcl_logits = self.model(xyz.permute(0,2,1)) # (B, 2, N), (B, num_class, N)
        pred = mcl_logits.softmax(dim=1)
        
        # ground truth
        gt          = batch["multi_label"] # (B, N)
        noise_gt    = batch["noise_label"] # (B, N)
        noise_mask  = noise_gt==0
        
        # end here during inference
        if split == 'inference': return xyz, pred, gt

        # meta classification
        prob1, prob2 = PairEnum(pred.permute(0,2,1), ~noise_mask)
        simi = Class2Simi(gt, 'hinge', ~noise_mask)
        mcl_loss = self.criterion(prob1, prob2, simi)

        # noise segmentation 
        noise_loss = self.noise_loss(noise_logits, noise_gt)

        # Compose loss
        loss = noise_loss * 0.5 + mcl_loss * 0.5

        # Metrics
        cluster = torch.argmax(pred, dim=1) + 1
        noise = torch.argmax(noise_logits.softmax(dim=1), dim=1)
        cluster[noise==0] = 0
        noise_iou, cluster_iou = self.acc(cluster, gt)

        # Logging
        self.log_value("loss", loss, split, B)
        self.log_value("noise_loss", noise_loss, split, B)
        self.log_value("mcl_loss", mcl_loss, split, B)
        self.log_value("Noise_IoU", noise_iou, split, B)
        self.log_value("Cluster_IoU", cluster_iou, split, B)

        # Visualisation
        if batch_idx == 0 and split == "valid":
            self.visualize_prediction(batch, cluster, gt, log=True)
        return loss

    def visualize_prediction(self, batch, segmentation_pred, segmentation_label, log=True):
        points = batch["point_clouds"].squeeze(0).cpu().numpy() # (N, 3)

        segmentation_pred = segmentation_pred.squeeze(0).cpu().numpy() # (N)
        segmentation_label = segmentation_label.squeeze(0).cpu().numpy() # (N)

        img_pred     = draw_scatterplot(points, seg_pred=segmentation_pred)
        img_gt       = draw_scatterplot(points, seg_gt=segmentation_label)
        if log: self.log_image(key='valid_pred', images=[img_pred])
        if log: self.log_image(key='valid_gt', images=[img_gt])
        return img_gt, img_pred, points

    def predict_step(self, batch, batch_idx):
        _, pred, gt = self(batch, batch_idx, 'inference')
        img_gt, img_pred, xyz = self.visualize_prediction(batch, pred, gt, log=False)
        img_pred = img_pred[...,::-1]
        img_gt = img_gt[...,::-1]
        noise_iou, cluster_iou = self.acc(pred, gt)
        return img_gt, img_pred, batch["plot_id"].squeeze(0).cpu().item(), xyz, gt, pred, noise_iou, cluster_iou

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay)
        scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: np.power(self.opt.learning_rate_decay, self.global_step)),
                'interval': 'step',
                'frequency': 1,
                'strict': True,
            }
        return [optimizer], [scheduler]
        

if __name__ == '__main__':
    import  torch
    model = Pointnet2Backbone(output_feature_dim=6)
    xyz = torch.rand(6, 1337, 3)

    end_points = {}
    end_points['point_clouds'] = xyz
    end_points = model(end_points)
    print(end_points['point_clouds_feat'].shape)

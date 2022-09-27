import torch
import torch.nn.functional as F
import numpy as np
from models.backbone_module import Pointnet2Backbone
import pointnet2.pointnet2_utils as pointnet2_utils
from pointnet2.pointnet2_modules import PointnetSAModuleMSG
from pointnet2.pointnet2_modules import PointnetSAModuleVotes
from pytorch_utils.module import LightningModule
from models.loss_helper import MCL
from torchmetrics import JaccardIndex
from utils.scatterplot import draw_scatterplot

def PairEnum(x):
    # x.shape (B, N, C)
    B = x.shape[0]
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 3, 'Input dimension must be 2'
    x1 = x.repeat(1, x.size(1), 1)
    x2 = x.repeat(1, 1, x.size(1)).view(B,-1, x.size(2))
    return x1,x2


def Class2Simi(x,mode='cls'):
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
    return out

class SegmentationModule(torch.nn.Module):
    def __init__(self, num_points, sampling, num_point_feat, max_clusters):
        super().__init__() 

        self.num_points = num_points
        self.sampling = sampling

        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_points,
                radius=0.3,
                nsample=16,
                mlp=[num_point_feat, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128, max_clusters+1,1)
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
        net = self.conv3(net) 

        return net

class MCLModel(torch.nn.Module):
    def __init__(self, num_points=1000, num_point_features=6, max_clusters=2) -> None:
        super().__init__()

        # Point Features
        if num_point_features > 0:
            self.backbone_feat = PointnetSAModuleMSG(npoint=num_point_features, radii=[0.6], nsamples=[16], mlps=[[0, num_points]])
        else:
            self.backbone_feat = None

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=num_point_features)

        self.pnet = SegmentationModule(num_points, "vote_fps", 256, max_clusters)

    def forward(self, inputs):
        end_points = inputs

        if self.backbone_feat:
            # generate features
            _, feat = self.backbone_feat(end_points['point_clouds'])
            end_points['point_clouds'] = torch.cat([end_points['point_clouds'], feat], dim=2) # concat point features to xyz

        end_points = self.backbone_net(end_points['point_clouds'], end_points)

        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']

        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))

        end_points = self.pnet(xyz, features, end_points)

        return end_points

class MCLModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super(MCLModule, self).__init__(*args, **kwargs)
        self.model = MCLModel(num_points=self.opt.num_points, num_point_features=self.opt.num_point_features, max_clusters=self.opt.max_cluster)
        self.criterion = MCL()
        self.iou = JaccardIndex(num_classes=self.opt.max_cluster+1, average=None)

    def forward(self, batch, batch_idx, split):
        B = batch["point_clouds"].shape[0]
        labels = batch["instance_labels"] # (B, N)
        simi = Class2Simi(labels)
        logits = self.model(batch)
        prob     = logits.softmax(dim=1) # (B, C, N)
        prob1, prob2 = PairEnum(prob.permute(0, 2, 1))
        iou = self.iou(torch.argmax(prob, dim=1), labels)
        loss = self.criterion(prob1, prob2, simi)

        self.log_value("loss", loss, split=split, batch_size=B)
        for i, acc in enumerate(iou):
            self.log_value("iou_{}".format("noise" if i==0 else "cluster{}".format(i)), acc, split=split, batch_size=B)
        if batch_idx == 0 and split == "valid":
            self.visualize_prediction(batch['point_clouds'], batch['center_label'], batch['bbox_dim'], prob, labels)
        return loss

    def visualize_prediction(self, points, gt_centers, dim, seg_pred, seg_gt, log=True):
        points = points.squeeze(0).cpu().numpy() # (N, 3)
        gt_centers = gt_centers.squeeze(0).cpu().numpy() # (2, 3)
        dim = dim.squeeze(0).cpu().numpy() # (2, 3)
        seg_pred = seg_pred.squeeze(0).cpu().softmax(dim=0) # (C, N)
        seg_pred = torch.argmax(seg_pred, dim=0).numpy() # (N)
        seg_gt = seg_gt.squeeze(0).cpu().numpy() # (N)
        
        bbox = np.concatenate([gt_centers, dim], axis=1)
        img_pred = draw_scatterplot(points, bbox=bbox, seg_pred=seg_pred)
        img_gt   = draw_scatterplot(points, bbox=bbox, seg_gt=seg_gt)
        
        if log: self.log_image(key='valid_pred', images=[img_pred])
        if log: self.log_image(key='valid_gt', images=[img_gt])

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
    num_points = 1000

    mcl_model = MCLModel(num_points=1000, num_point_features=0, max_clusters=2).cuda()

    inputs = {}
    inputs['point_clouds'] = torch.randn((8, num_points, 3)).cuda()
    seg = mcl_model(inputs)
    print(seg.shape)
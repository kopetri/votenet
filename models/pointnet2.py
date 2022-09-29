from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_utils.module import LightningModule
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation


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

class ClusterSeparation(Pointnet2Backbone):
    def __init__(self, num_classes, *args, **kwargs):
        super(ClusterSeparation, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

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
        return x

class ClusterSeparationModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = ClusterSeparation(self.opt.max_clusters+1)
        self.criterion = torch.nn.CrossEntropyLoss(torch.tensor([0.9, 0.1]))

    def forward(self, batch, batch_idx, split):
        xyz = batch['point_clouds'] # (B, N. 3)
        pred = self.model(xyz) # (B, num_class, N)
        gt = batch["noise_labels"] # (B, N)
        B = xyz.shape[0]
        loss = self.criterion(pred, gt)
        self.log_value("loss", loss, split, B)
        return loss
        

if __name__ == '__main__':
    import  torch
    model = Pointnet2Backbone(output_feature_dim=6)
    xyz = torch.rand(6, 1337, 3)

    end_points = {}
    end_points['point_clouds'] = xyz
    end_points = model(end_points)
    print(end_points['point_clouds_feat'].shape)
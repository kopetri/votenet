# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from utils.pc_util import random_sampling, rotz, rotate_aligned_boxes

def generate_splits(directory, train=0.8, valid=0.05, test=0.15):
    print("Generating splits...")
    def __write_files__(files, out):
        if isinstance(out, Path):
            out = out.as_posix()
        with open(out, "w") as txtfile:
            for f in files:
                txtfile.write("{}\n".format(f.stem))

    files = [f for f in directory.glob("*") if f.suffix == ".npy"]

    N = len(files)
    n_train = int(N * train)
    n_valid = int(N * valid)
    n_test  = int(N * test)

    rest = N - (n_train + n_valid + n_test)
    n_test += rest

    assert n_train + n_valid + n_test == N, "{} !+ {}".format(n_train + n_valid + n_test, N)

    np.random.shuffle(files)

    train_split = files[0:n_train]
    valid_split = files[n_train:n_train+n_valid]
    test_split  = files[n_train+n_valid:n_train+n_valid+n_test]

    __write_files__(train_split, Path(directory, "train.txt"))
    __write_files__(valid_split, Path(directory, "valid.txt"))
    __write_files__(test_split,  Path(directory, "test.txt"))

def get_max_cluster_count(directory):
    from tqdm import tqdm
    files = [f for f in directory.glob("*") if f.suffix == ".npy"]
    max_cluster = 0
    numbers = []
    for file in tqdm(files):
        data = np.load(file)
        label = data[:, 2]
        n = int(np.max(label) + 1)
        numbers.append(n)
        if n > 10:
            file.unlink()
        max_cluster = max(n, max_cluster)
    print("Maximum number of cluster is: ", max_cluster)
    print(np.unique(np.array(numbers), return_counts=True))


MAX_NUM_OBJ = 2

class ClusterSeparatonDatasetConfig(object):
    def __init__(self):
        self.num_class = 2
        self.num_heading_bin = 1
        self.num_size_cluster = 2
        self.mean_size_arr = np.array([
            [1.1498247, 1.1521318, 0.],
            [1.1353437, 1.1349982, 0.]
        ])
        
class RealClusterDataset(Dataset):
    def __init__(self, path, split='train', augment=False) -> None:
        super().__init__()
        self.use_small = "small" in split
        self.split = split.replace("_small", "")
        self.path = Path(path)
        self.augment = augment
        self.ids = self.load_split()
        if self.use_small: print("DEBUG using small version!")
        print("Found {} scatterplots for split {}".format(len(self.ids), split))

    def load_split(self):
        split_file = self.path/"{}.txt".format(self.split)
        if not split_file.exists():
            generate_splits(self.path)
        with open(split_file, "r") as splitfile:
            lines = [s.strip() for s in splitfile.readlines()]
            if self.use_small: lines = lines[0:50]
            return lines

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):        
        plot_id = self.ids[idx]        
        data = np.load(self.path/'{}.npy'.format(plot_id)) # X,Y,Label

        point_cloud = np.zeros((data.shape[0], 3))
        point_cloud[:, 0:2] = data[:,0:2]
        multi_label = data[:, 2] + 1 # 0 noise, 1 cluster 1, 2 cluster 2, ...
        noise_label = np.clip(data[:, 2] + 1, 0, 1) # 0 noise, 1 cluster

     
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
            
            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            
        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['multi_label'] = multi_label.astype(np.int64)
        ret_dict['noise_label'] = noise_label.astype(np.int64)
        ret_dict['plot_id'] = np.array(plot_id).astype(np.int64)
        return ret_dict


class ClusterSeparationDataset(Dataset):
       
    def __init__(self, path, split='train', num_points=5000, use_color=False, use_height=False, augment=False, static_choice=False):
        self.use_small = "small" in split
        self.split = split.replace("_small", "")
        self.path = Path(path)
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.augment = augment
        self.ids = self.load_split()
        self.mean_size_arr = self.compute_mean_size_arr()
        if self.use_small: print("DEBUG using small version!")
        print("Found {} scatterplots for split {}".format(len(self.ids), split))
        print("Mean size arr: ", self.mean_size_arr)
        self.static_choice = static_choice

    def load_split(self):
        with open(self.path/"{}.txt".format(self.split), "r") as splitfile:
            lines = [s.strip() for s in splitfile.readlines()]
            if self.use_small: lines = lines[0:50]
            return lines
        
    def compute_mean_size_arr(self):
        size_arr_a = []
        size_arr_b = []
        for idx in range(self.__len__()):
            bboxs = np.load(self.path/'{}_bbox.npy'.format(self.ids[idx]))
            for bbox in bboxs:
                if bbox[-1] == 0:
                    size_arr_a.append(bbox[3:6])
                else:
                    size_arr_b.append(bbox[3:6])
        return np.array([np.mean(size_arr_a, axis=0), np.mean(size_arr_b, axis=0)])
       
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            angle_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_votes: (N,3) with votes XYZ
            point_votes_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            pcl_color: unused
        """
        
        plot_id = self.ids[idx]        
        vertices        = np.load(self.path/'{}_vertex.npy'.format(plot_id)) # X,Y,1
        instance_labels = np.load(self.path/'{}_ins_id.npy'.format(plot_id)) # 0 non cluster, 1, 2
        semantic_labels = np.load(self.path/'{}_sem_id.npy'.format(plot_id)) # 0 -> non cluster, 1 -> cluster
        instance_bboxes = np.load(self.path/'{}_bbox.npy'.format(plot_id))

        # votes = tuple(mask, bbox)

        point_cloud = vertices[:,0:3]
   
            
        # ------------------------------- LABELS ------------------------------        
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        
        if self.static_choice:
            N = len(point_cloud)
            step = N // self.num_points
            choices = np.arange(0,N,step)
            assert len(choices) == self.num_points, len(choices)
            point_cloud = point_cloud[choices]
        else:
            point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
                
        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0],:] = instance_bboxes[:,0:6]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                target_bboxes[:,1] = -1 * target_bboxes[:,1]                                
            
            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label            
            x = point_cloud[ind,:3]
            center = 0.5*(x.min(0) + x.max(0))
            point_votes[ind, :] = center - x
            point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 

        class_ind = [x for x in instance_bboxes[:,-1]]   
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6] - self.mean_size_arr
            
        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['noise_label'] = semantic_labels.astype(np.int64)
        ret_dict['multi_label'] = instance_labels.astype(np.int64)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['bbox_dim'] = target_bboxes.astype(np.float32)[:,3:6]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))                                
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
            [x for x in instance_bboxes[:,-1][0:instance_bboxes.shape[0]]]                
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['plot_id'] = np.array(plot_id).astype(np.int64)
        #ret_dict['pcl_color'] = pcl_color
        return ret_dict
        
############# Visualizaion ########

def viz_votes(pc, point_votes, point_votes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]    
    pc_util.write_ply(pc_obj, 'pc_obj{}.ply'.format(name))
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1{}.ply'.format(name))
    
def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals, name=''):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i,0:3]
        heading_angle = 0 # hard code to 0
        box_size = DC.mean_size_arr[size_classes[i], :] + size_residuals[i, :]
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)        
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs{}.ply'.format(name))
    pc_util.write_ply(label[mask==1,:], 'gt_centroids{}.ply'.format(name))

    
if __name__=='__main__': 
    import cv2
    from utils.scatterplot import draw_scatterplot
    dataset = RealClusterDataset(path="../dataset/sebi_onze_dataset/datasets/data-gov-dataset/no_project/", split="train", augment=False)
    for d in dataset:
        points = d["point_clouds"] # (N, 3)
        print(points.shape)
        break

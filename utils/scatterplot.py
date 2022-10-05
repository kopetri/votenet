import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import torch
import cv2

def get_n_colors(N):
    x = np.linspace(0.0, 1.0, N)
    colors = plt.colormaps['prism'](x)[:, :3]
    colors[0, :] = 0.0
    return colors


def draw_scatterplot(points=None, sem=None, instance=None, bbox=None, pred=None, seg_pred=None, seg_gt=None, objectness_score=None, objectness_label=None, num_proposal=None, near=None, far=None):
    colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green'}
    colors_sem = {0:'tab:purple', 1:'tab:blue'}
    colors = get_n_colors(num_proposal if num_proposal else 11)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.axis('off')
        
    
    if not points is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=2)
    if not sem is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=10, color=[colors_sem[i] for i in sem])
    if not instance is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=2, color=[colors[i] for i in instance])
    #if not seg_pred is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=25, c=seg_pred, cmap=new_cmap, vmin=0, vmax=100)
    #if not seg_gt is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=25, c=seg_gt, cmap=new_cmap, vmin=0, vmax=100)
    if not seg_pred is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=25, c=[colors[i] for i in seg_pred])
    if not seg_gt is None:   plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=25, c=[colors[i] for i in seg_gt])
    
    if not bbox is None:
        for b in bbox:
            if b[3] == 0:continue
            plt.plot(b[0], b[1], 'go')
            if near is not None: ax.add_patch(Circle(b[0:2], radius=near, fill=False))
            if far is not None: ax.add_patch(Circle(b[0:2], radius=far, fill=False))
            ax.add_patch(Rectangle((b[0]-b[3]*0.5,b[1]-b[4]*0.5), b[3], b[4], linewidth=1, edgecolor='g', facecolor='none'))

    if not pred is None:
        for i,c in enumerate(pred):
            if objectness_score is not None and objectness_score[i]:
                plt.plot(c[0], c[1], 'bo')
            elif objectness_label is not None and objectness_label[i]:
                plt.plot(c[0], c[1], 'b^')
            else:
                plt.plot(c[0], c[1], 'bx')
    
    
    fig.tight_layout(pad=0)


    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image_from_plot


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in np.arange(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

def extract_clusters(matrix):
    K = matrix.shape[0]
    matrix = torch.triu(matrix, diagonal=1)
    matrix.fill_diagonal_(1)
    clusters = []
    finished = []
    for i in range(K):
      if i in finished:continue
      current_nodes = []
      row = matrix[i]
      for j in range(K):   
        if row[j] > 0.5:
          current_nodes.append([j, row[j].item()])
          finished.append(j)
      clusters.append(current_nodes)

    return clusters

def make_adjacent_matrix(vec):
    K = len(vec)
    matrix = torch.zeros((K, K))
    for i in torch.unique(vec):
      combs = torch.combinations(torch.where(vec==i)[0], with_replacement=True).permute(1,0)
      combs_inverse = torch.flip(combs, [0])

      matrix[combs.tolist()] = 1
      matrix[combs_inverse.tolist()] = 1
    return matrix

def get_ambiguous_item(i, clusters):
    result = [(idx, score) for c in clusters for idx,score in c if idx == i]
    if len(result) > 1:
      return np.max([score for _, score in result])
    else:
      return -1

def invalidate(score, idx, clusters):
    return [[[i,-1] if (idx==i and s<score) else [i,s] for (i,s) in c] for c in clusters]
    

def clean_clusters(clusters, K):
    for i in range(K):
      score = get_ambiguous_item(i, clusters)
      if score > 0:
        clusters = invalidate(score, i, clusters)

    clusters = [[i for (i,s) in c if not s==-1] for c in clusters]
    return clusters

def clusters2map(clusters, K):
    cluster_map = torch.zeros(K)
    for idx, c in enumerate(clusters):
      for i in c:
        cluster_map[i] = idx

    return cluster_map

def adjacent_matrix_to_cluster(matrix):
    if len(matrix.shape) == 2:
        K = matrix.shape[1]
        clusters = extract_clusters(matrix)
        clusters = clean_clusters(clusters=clusters, K=K)
        return clusters2map(clusters=clusters, K=K).to(matrix)
    elif len(matrix.shape) == 4: # (B, 2, K, K)
        K = matrix.shape[2]
        matrix = torch.argmax(matrix, dim=1)
        cluster_map = torch.zeros((matrix.shape[0], K))
        for bidx, m in enumerate(matrix):
            clusters = extract_clusters(m)
            clusters = clean_clusters(clusters=clusters, K=K)
            cluster_map[bidx] = clusters2map(clusters=clusters, K=K)
        return cluster_map.to(matrix)
    else:
        raise ValueError(matrix.shape)


if __name__ == '__main__':
    def nn_dist(a, b):
        M = a.shape[0]
        N = b.shape[0]

        X = np.expand_dims(a, 0).repeat(N, axis=0)
        Y = np.expand_dims(b, 1).repeat(M, axis=1)
        dist = X - Y 
        dist = np.sum(dist**2, axis=-1)
        dist1,dist2 = np.min(dist, axis=0), np.min(dist,axis=1)
        return dist1, dist2
        
    import cv2
    points = np.load("H:/data/sebi_onze_dataset/datasets/Onze/00000_vertex.npy")
    bbox = np.load("H:/data/sebi_onze_dataset/datasets/Onze/00000_bbox.npy")

    K = 64
    NEAR_THRESHOLD = 0.2
    FAR_THRESHOLD = 0.9
    pred = np.random.uniform(-1,1,(K,2))


    dist1, dist2 = nn_dist(pred[:, 0:2], bbox[:, 0:2])
    
    euclidean_dist1 = np.sqrt(dist1+1e-6)
    objectness_label = np.zeros(K, dtype=int)
    objectness_mask = np.zeros(K)
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    print(objectness_label)

    img = draw_scatterplot(points, pred=pred, bbox=bbox, objectness_label=objectness_label, near=NEAR_THRESHOLD, far=FAR_THRESHOLD)
    img = img[..., ::-1] # rgb -> bgr
    #print(img.shape)
    cv2.imshow("scatterplot", img)
    cv2.waitKey(0)
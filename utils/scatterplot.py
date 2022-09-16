import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def draw_scatterplot(points=None, sem=None, instance=None, bbox=None, pred=None, seg_pred=None, seg_gt=None, objectness_score=None):
    colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green'}
    colors_sem = {0:'tab:purple', 1:'tab:cyan'}
    new_cmap = rand_cmap(256, type='bright', first_color_black=True, last_color_black=False, verbose=False)
    fig, ax = plt.subplots()
    plt.axis('off')
        
    
    if not points is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=2)
    if not sem is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=10, color=[colors_sem[i] for i in sem])
    if not instance is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=2, color=[colors[i] for i in instance])
    if not seg_pred is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=50, c=seg_pred, cmap=new_cmap, vmin=0, vmax=100)
    if not seg_gt is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=25, c=seg_gt, cmap=new_cmap, vmin=0, vmax=100)
    
    if not bbox is None:
        for b in bbox:
            if b[3] == 0:continue
            plt.plot(b[0], b[1], 'go')
            ax.add_patch(Rectangle((b[0]-b[3]*0.5,b[1]-b[4]*0.5), b[3], b[4], linewidth=1, edgecolor='g', facecolor='none'))

    if not pred is None:
        for i,c in enumerate(pred):
            if objectness_score is None or objectness_score[i]:
                plt.plot(c[0], c[1], 'ro')
            else:
                plt.plot(c[0], c[1], 'rx')
    
    
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


if __name__ == '__main__':
    import cv2
    points = np.random.normal(0,1, (100,3))

    img = draw_scatterplot(points)

    print(img.shape)
    cv2.imshow("scatterplot", img)
    cv2.waitKey(0)
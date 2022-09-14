import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def draw_scatterplot(points, sem=None, instance=None, bbox=None, pred=None):
    colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green'}
    colors_sem = {0:'tab:purple', 1:'tab:cyan'}
    fig, ax = plt.subplots()
    plt.axis('off')
        
    
    plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=2)
    if not sem is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=10, color=[colors_sem[i] for i in sem])
    if not instance is None: plt.scatter(x=points[:,0], y=points[:,1], marker='o', s=2, color=[colors[i] for i in instance])
    
    if not bbox is None:
        for b in bbox:
            if b[3] == 0:continue
            plt.plot(b[0], b[1], 'go')
            ax.add_patch(Rectangle((b[0]-b[3]*0.5,b[1]-b[4]*0.5), b[3], b[4], linewidth=1, edgecolor='g', facecolor='none'))

    if not pred is None:
        for c in pred:
            plt.plot(c[0], c[1], 'ro')
    
    
    fig.tight_layout(pad=0)


    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot


if __name__ == '__main__':
    import cv2
    points = np.random.normal(0,1, (100,3))

    img = draw_scatterplot(points)

    print(img.shape)
    cv2.imshow("scatterplot", img)
    cv2.waitKey(0)
import matplotlib.pyplot as plt
import numpy as np

def draw_adjacent_matrix(matrix):
    fig, ax = plt.subplots()
    plt.axis('off')
    plt.matshow(matrix, cmap='Blues')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
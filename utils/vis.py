import cv2
import numpy as np

def draw_adjacent_matrix(matrix, round=True, width=256, height=256):
    if round:
        matrix = np.round(matrix, 0)
    matrix = np.repeat(matrix, height, axis=0)
    matrix = np.repeat(matrix, width, axis=1)

    matrix *= 255
    matrix = np.clip(matrix, 0,255)
    matrix = matrix.astype(np.uint8)

    matrix = cv2.applyColorMap(matrix, cv2.COLORMAP_BONE)
    return cv2.resize(matrix, (width, height))[...,::-1]
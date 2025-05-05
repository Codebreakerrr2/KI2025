import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import time
import random

def diffuse_smell(grid, target, iterations=2000):  # Mehr Iterationen
    rows, cols = max(grid.nodes())[0] + 1, max(grid.nodes())[1] + 1
    kernel = np.array([[0.05, 0.2, 0.05],  # Stärkerer Diffusionskernel
                       [0.2, 0.2, 0.2],
                       [0.05, 0.2, 0.05]])

    concentration = np.zeros((rows, cols))
    concentration[target] = 1.0

    for _ in range(iterations):
        concentration = convolve(concentration, kernel, mode='constant')


    # Skaliere auf [0,1]
    concentration /= concentration.max()

    for (x, y) in grid.nodes():
        grid.nodes[(x, y)]['smell'] = concentration[x, y]
    return grid

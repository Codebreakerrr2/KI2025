import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# --- Dein Code ---
def create_grid_graph(rows, cols):
    G = nx.grid_2d_graph(rows, cols)
    for u, v in G.edges():
        G.edges[u, v]['weight'] = random.random()
    return G

def get_kruskal_maze(G):
    return nx.minimum_spanning_tree(G, algorithm='kruskal')

def diffuse_smell(grid, target, iterations=50):
    rows, cols = max(grid.nodes())[0] + 1, max(grid.nodes())[1] + 1
    kernel = np.array([[0.05, 0.2, 0.05],
                       [0.2,  0.2, 0.2],
                       [0.05, 0.2, 0.05]])

    concentration = np.zeros((rows, cols))
    concentration[target] = 1.0

    for _ in range(iterations):
        concentration = convolve(concentration, kernel, mode='constant')

    concentration /= concentration.max()

    for (x, y) in grid.nodes():
        grid.nodes[(x, y)]['smell'] = concentration[x, y]
    return grid

# --- Neu: Heatmap-Plot ---
def plot_smell_heatmap(grid):
    rows, cols = max(grid.nodes())[0] + 1, max(grid.nodes())[1] + 1
    smell = np.zeros((rows, cols))

    for (x, y) in grid.nodes():
        smell[x, y] = grid.nodes[(x, y)]['smell']

    plt.figure(figsize=(8, 6))
    plt.imshow(smell, cmap='hot', origin='lower')
    plt.colorbar(label='Geruchsstärke')
    plt.title('Heatmap der Geruchskonzentration')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# --- Anwendung ---
rows, cols = 20, 20
G = create_grid_graph(rows, cols)
maze = get_kruskal_maze(G)
target = (rows //2, cols // 5)  # Target in der Mitte
maze = diffuse_smell(maze, target, iterations=100)
plot_smell_heatmap(maze)

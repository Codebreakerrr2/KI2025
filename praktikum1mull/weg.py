import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.ndimage import convolve

# 1. Maze-Erstellung
def create_grid_graph(rows, cols):
    G = nx.grid_2d_graph(rows, cols)
    for u, v in G.edges():
        G.edges[u, v]['weight'] = random.random()
    return G

def get_kruskal_maze(G):
    return nx.minimum_spanning_tree(G, algorithm='kruskal')

def add_shortcuts(G, num_extra_edges=3):
    nodes = list(G.nodes())
    possible_edges = [(u, v) for u in nodes for v in nodes
                     if u != v and not G.has_edge(u, v)]
    for u, v in random.sample(possible_edges, min(num_extra_edges, len(possible_edges))):
        G.add_edge(u, v, weight=random.random())
    return G

# 2. Geruchsverteilung
def diffuse_smell(grid, target, iterations=2000):
    rows, cols = max(grid.nodes())[0] + 1, max(grid.nodes())[1] + 1
    kernel = np.array([[0.05, 0.2, 0.05],
                       [0.2, 0.2, 0.2],
                       [0.05, 0.2, 0.05]])
    concentration = np.zeros((rows, cols))
    concentration[target] = 1.0
    for _ in range(iterations):
        concentration = convolve(concentration, kernel, mode='constant')
    concentration /= concentration.max()
    for (x, y) in grid.nodes():
        grid.nodes[(x, y)]['smell'] = concentration[x, y]
    return grid

# 3. Agenten
def astar_smell_agent(G, start, target):
    t_start = time.time()
    heuristic = lambda u, v: 1.0 - G.nodes[u]['smell']
    path = nx.astar_path(G, start, target, heuristic=heuristic, weight='weight')
    steps = len(path)
    return path, time.time() - t_start, steps

def greedy_smell_agent(G, start, target):
    t_start = time.time()
    path = [start]
    current = start
    visited = set()
    steps = 0
    while current != target:
        visited.add(current)
        neighbors = [n for n in G.neighbors(current) if n not in visited]
        steps += 1
        if not neighbors:
            if len(path) > 1:
                current = path[-2]
                path.pop()
            else:
                break
        else:
            next_node = max(neighbors, key=lambda n: G.nodes[n]['smell'])
            path.append(next_node)
            current = next_node
    return path, time.time() - t_start, steps

def dfs_agent(G, start, target):
    t_start = time.time()
    stack = [(start, [start])]
    visited = set()
    steps = 0
    while stack:
        node, path = stack.pop()
        steps += 1
        if node == target:
            return path, time.time() - t_start, steps
        if node not in visited:
            visited.add(node)
            for n in G.neighbors(node):
                stack.append((n, path + [n]))
    return [], time.time() - t_start, steps

# 4. Visualisierung
def draw_maze_with_paths(G, paths, start, target):
    pos = {(x, y): (x, y) for x, y in G.nodes()}  #

    plt.figure(figsize=(15, 8))
    # Maze
    nx.draw(G, pos, node_size=10, edge_color='lightgray', width=1)

    colors = {'DFS': 'blue', 'Greedy': 'green', 'A*': 'red'}
    for algo, path in paths.items():
        edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=2.5, edge_color=colors[algo], label=algo)

    # Start und Ziel hervorheben
    nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='gold', node_size=80, label='Start')
    nx.draw_networkx_nodes(G, pos, nodelist=[target], node_color='purple', node_size=80, label='Ziel')

    plt.legend()
    plt.axis('off')
    plt.title("Maze mit Agentenpfaden (DFS, Greedy, A*)")
    plt.show()

# Hauptlauf
rows, cols = 20, 20
num_shortcuts = 5

grid = create_grid_graph(rows, cols)
kruskal_maze = get_kruskal_maze(grid)
final_maze = add_shortcuts(kruskal_maze.copy(), num_shortcuts)

start = (0, 0)
target = (random.randint(0, rows-1), random.randint(0, cols-1))

final_maze = diffuse_smell(final_maze, target)

# Pfade der Agenten
path_dfs, _, _ = dfs_agent(final_maze, start, target)
path_greedy, _, _ = greedy_smell_agent(final_maze, start, target)
path_astar, _, _ = astar_smell_agent(final_maze, start, target)

paths = {
    'DFS': path_dfs,
    'Greedy': path_greedy,
    'A*': path_astar
}

# Zeichnen
draw_maze_with_paths(final_maze, paths, start, target)

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import time
import random


# 1. Grid erstellen
def create_grid_graph(rows, cols):
    G = nx.grid_2d_graph(rows, cols)
    # Zufällige Gewichte für Kruskal zuweisen
    for u, v in G.edges():

        G.edges[u, v]['weight'] = random.random()
    return G

# 2. Spannbaum mit Kruskal
def get_kruskal_maze(G):
    return nx.minimum_spanning_tree(G, algorithm='kruskal')

# 3. Zufällige Kanten hinzufügen
def add_shortcuts(G, num_extra_edges=3):
    nodes = list(G.nodes())
    possible_edges = [(u, v) for u in nodes for v in nodes
                     if u != v and not G.has_edge(u, v)]
    for u, v in random.sample(possible_edges, min(num_extra_edges, len(possible_edges))):
        G.add_edge(u, v, weight=random.random())
    return G

def add_traps(G,num_traps = None, trap_damage):
    if num_traps is None:
        num_traps = len(G.nodes())//4
    trap_nodes = random.sample(list(G.nodes), min(num_traps,len(G.nodes())))
    for node in trap_nodes:
        G.nodes[node]['trap'] = True
        G.nodes[node]['damage'] = trap_damage
    return G


'Praktikum1'

#
# rows, cols = 20, 20
# num_shortcuts = 5  # Anzahl zusätzlicher Pfade
#
# # Maze generieren
# grid = create_grid_graph(rows, cols)
# kruskal_maze = get_kruskal_maze(grid)
# final_maze = add_shortcuts(kruskal_maze.copy(), num_shortcuts)

# # Visualisierung
# pos = {(x, y): (x, y) for x, y in grid.nodes()}
# plt.figure(figsize=(12, 4))
#
# # Originales Grid
# plt.subplot(131)
# plt.title("Originales Grid")
# nx.draw(grid, pos, node_size=0, edge_color='gray', width=1)
#
# # Kruskal-Maze (ohne Zyklen)
# plt.subplot(132)
# plt.title("Kruskal-Maze (reiner Spannbaum)")
# nx.draw(kruskal_maze, pos, node_size=20, edge_color='blue', width=2)
#
# # Mit Shortcuts
# plt.subplot(133)
# plt.title(f"Mit {num_shortcuts} Shortcuts")
# nx.draw(final_maze, pos, node_size=20, edge_color='red', width=1.5)
#
# plt.tight_layout()
# plt.show()
# 2. Verbesserte Geruchsdiffusion





#maze generieren

def plot_maze_and_paths(maze, paths, labels, colors, start, target):
    pos = {(x, y): (y, -x) for x, y in maze.nodes()}

    plt.figure(figsize=(18, 6))

    for i, (path, label, color) in enumerate(zip(paths, labels, colors)):
        plt.subplot(1, 3, i + 1)
        plt.title(label)

        # Nur das Maze zeichnen
        nx.draw(maze, pos, node_size=5, edge_color='lightgray', width=0.5)

        # Pfad zeichnen
        if len(path) > 1:
            edge_list = list(zip(path, path[1:]))
            nx.draw_networkx_edges(maze, pos, edgelist=edge_list, edge_color=color, width=3)

        # Pfad-Knoten
        nx.draw_networkx_nodes(maze, pos, nodelist=path, node_color=color, node_size=20)

        # Start- und Zielknoten
        nx.draw_networkx_nodes(maze, pos, nodelist=[start], node_color='lime', node_size=100, label='Start')
        nx.draw_networkx_nodes(maze, pos, nodelist=[target], node_color='red', node_size=100, label='Ziel')

    plt.tight_layout()
    plt.show()

'Praktikum1'
# ---  Hauptcode ---
# Maze generieren
# maze1 = get_kruskal_maze(create_grid_graph(50, 50))
# maze = add_shortcuts(maze1,10)
#
#
# # Diffuse Smell für Greedy und A* Agenten
# maze = diffuse_smell(maze, (19, 19), iterations=50)

# Agents aufrufen
# pfadeDFS, zeitDFS, schritteDFS = dfs_agent(maze, (0, 0), (19, 19))
# pfadeGreedy, zeitGreedy, schritteGreedy = greedy_smell_agent(maze, (0, 0), (19, 19))
# pfadeStar, zeitStar, schritteStar = astar_smell_agent(maze, (0, 0), (19, 19))
#
# # Ergebnisse ausgeben
# print("=== Ergebnisse ===")
# print(f"DFS:     Länge des Pfades: {len(pfadeDFS)}, Zeit: {zeitDFS:.4f}s, Schritte: {schritteDFS}")
# print(f"Greedy:  Länge des Pfades: {len(pfadeGreedy)}, Zeit: {zeitGreedy:.4f}s, Schritte: {schritteGreedy}")
# print(f"A*:      Länge des Pfades: {len(pfadeStar)}, Zeit: {zeitStar:.4f}s, Schritte: {schritteStar}")
# # Vergleiche als Balkendiagramme
# def plot_comparison(pfade_längen, zeiten, schritte, labels):
#     x = np.arange(len(labels))  # Balkenpositionen
#     width = 0.25  # Breite der Balken
#
#     fig, axs = plt.subplots(1, 3, figsize=(18, 5))
#
#     # Plot 1: Pfadlängen
#     axs[0].bar(x, pfade_längen, width, color=['blue', 'orange', 'green'])
#     axs[0].set_title('Pfadlänge')
#     axs[0].set_xticks(x)
#     axs[0].set_xticklabels(labels)
#     axs[0].set_ylabel('Knoten')
#
#     # Plot 2: Schritte
#     axs[1].bar(x, schritte, width, color=['blue', 'orange', 'green'])
#     axs[1].set_title('Anzahl Schritte')
#     axs[1].set_xticks(x)
#     axs[1].set_xticklabels(labels)
#     axs[1].set_ylabel('Anzahl')
#
#     # Plot 3: Zeiten
#     axs[2].bar(x, zeiten, width, color=['blue', 'orange', 'green'])
#     axs[2].set_title('Benötigte Zeit')
#     axs[2].set_xticks(x)
#     axs[2].set_xticklabels(labels)
#     axs[2].set_ylabel('Sekunden')
#
#     plt.tight_layout()
#     plt.show()


# # Werte vorbereiten
# pfade_längen = [len(pfadeDFS), len(pfadeGreedy), len(pfadeStar)]
# zeiten = [zeitDFS, zeitGreedy, zeitStar]
# schritte = [schritteDFS, schritteGreedy, schritteStar]
# labels = ['DFS', 'Greedy', 'A*']
#
# # Plot aufrufen
# plot_comparison(pfade_längen, zeiten, schritte, labels)
# def run_multiple_iterations(agent_function, maze, start, target, iterations=10):
#     pfade = []
#     zeiten = []
#     schritte = []
#
#     for _ in range(iterations):
#         path, time_taken, steps = agent_function(maze, start, target)
#         pfade.append(len(path))
#         zeiten.append(time_taken)
#         schritte.append(steps)
#
#     # Durchschnitt berechnen
#     avg_path_length = np.mean(pfade)
#     avg_time = np.mean(zeiten)
#     avg_steps = np.mean(schritte)
#
#     return avg_path_length, avg_time, avg_steps
#
#
# # 10000 Iterationen für jeden Agenten
# avg_pfadDFS, avg_zeitDFS, avg_schritteDFS = run_multiple_iterations(dfs_agent, maze, (0, 0), (19, 19), iterations=10)
# avg_pfadGreedy, avg_zeitGreedy, avg_schritteGreedy = run_multiple_iterations(greedy_smell_agent, maze, (0, 0), (19, 19), iterations=10)
# avg_pfadStar, avg_zeitStar, avg_schritteStar = run_multiple_iterations(astar_smell_agent, maze, (0, 0), (19, 19), iterations=10)
#
# # Ergebnisse ausgeben
# print("=== Durchschnittliche Ergebnisse (über 10000 Iterationen) ===")
# print(f"DFS:     Durchschnittliche Pfadlänge: {avg_pfadDFS:.2f}, Durchschnittliche Zeit: {avg_zeitDFS:.4f}s, Durchschnittliche Schritte: {avg_schritteDFS}")
# print(f"Greedy:  Durchschnittliche Pfadlänge: {avg_pfadGreedy:.2f}, Durchschnittliche Zeit: {avg_zeitGreedy:.4f}s, Durchschnittliche Schritte: {avg_schritteGreedy}")
# print(f"A*:      Durchschnittliche Pfadlänge: {avg_pfadStar:.2f}, Durchschnittliche Zeit: {avg_zeitStar:.4f}s, Durchschnittliche Schritte: {avg_schritteStar}")

# Vergleichsdiagramme
# def plot_comparison_avg(pfade_längen, zeiten, schritte, labels):
#     x = np.arange(len(labels))  # Balkenpositionen
#     width = 0.25  # Breite der Balken
#
#     fig, axs = plt.subplots(1, 3, figsize=(18, 5))
#
#     # Plot 1: Durchschnittliche Pfadlängen
#     axs[0].bar(x, pfade_längen, width, color=['blue', 'orange', 'green'])
#     axs[0].set_title('Durchschnittliche Pfadlänge')
#     axs[0].set_xticks(x)
#     axs[0].set_xticklabels(labels)
#     axs[0].set_ylabel('Knoten')
#
#     # Plot 2: Durchschnittliche Schritte
#     axs[1].bar(x, schritte, width, color=['blue', 'orange', 'green'])
#     axs[1].set_title('Durchschnittliche Anzahl Schritte')
#     axs[1].set_xticks(x)
#     axs[1].set_xticklabels(labels)
#     axs[1].set_ylabel('Anzahl')
#
#     # Plot 3: Durchschnittliche Zeiten
#     axs[2].bar(x, zeiten, width, color=['blue', 'orange', 'green'])
#     axs[2].set_title('Durchschnittliche benötigte Zeit')
#     axs[2].set_xticks(x)
#     axs[2].set_xticklabels(labels)
#     axs[2].set_ylabel('Sekunden')
#
#     plt.tight_layout()
#     plt.show()
#
# # Werte für den Durchschnitt vorbereiten
# pfade_längen_avg = [avg_pfadDFS, avg_pfadGreedy, avg_pfadStar]
# zeiten_avg = [avg_zeitDFS, avg_zeitGreedy, avg_zeitStar]
# schritte_avg = [avg_schritteDFS, avg_schritteGreedy, avg_schritteStar]
# labels_avg = ['DFS', 'Greedy', 'A*']
#
# # Durchschnittsvergleichs-Plot aufrufen
# plot_comparison_avg(pfade_längen_avg, zeiten_avg, schritte_avg, labels_avg)

'Praktikum1'

# def run_random_target_experiment(agent_fn, maze, start, nodes, iterations=10000):
#     total_path, total_time, total_steps = 0, 0.0, 0
#     for _ in range(iterations):
#         target = random.choice(nodes)
#         path, t, steps = agent_fn(maze, start, target)
#         total_path  += len(path)
#         total_time  += t
#         total_steps += steps
#     return total_path/iterations, total_time/iterations, total_steps/iterations
#
# # --- Anwendung ---
# nodes = list(maze.nodes())
# start = (0, 0)
#
# avg_pdfs, avg_tdfs, avg_sdfs = run_random_target_experiment(dfs_agent,        maze, start, nodes)
# avg_pgre, avg_tgre, avg_sgre = run_random_target_experiment(greedy_smell_agent, maze, start, nodes)
# avg_past, avg_tast, avg_sast = run_random_target_experiment(astar_smell_agent,  maze, start, nodes)
#
# print("=== Durchschnitt über zufällige Targets (10 000 Iterationen) ===")
# print(f"DFS:     Pfadlänge={avg_pdfs:.2f}, Zeit={avg_tdfs:.4f}s, Schritte={avg_sdfs:.1f}")
# print(f"Greedy:  Pfadlänge={avg_pgre:.2f}, Zeit={avg_tgre:.4f}s, Schritte={avg_sgre:.1f}")
# print(f"A*:      Pfadlänge={avg_past:.2f}, Zeit={avg_tast:.4f}s, Schritte={avg_sast:.1f}")

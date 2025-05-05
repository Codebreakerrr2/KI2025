import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import time
import random


def astar_smell_agent(G, start, target):
    t_start = time.time()
    heuristic = lambda u, v: 1.0 - G.nodes[u]['smell']
    path = nx.astar_path(G, start, target, heuristic=heuristic, weight='weight')
    steps = len(path)  # A* läuft normalerweise perfekt → Schritte = Länge Pfad
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
        steps += 1  # Bewegung zählt als Schritt

        if not neighbors:  # Sackgasse → Backtracking
            if len(path) > 1:
                current = path[-2]
                path.pop()
            else:
                break  # Kein Weg
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
        steps += 1  # Jeder Knoten, den wir anschauen, ist ein Schritt
        if node == target:
            return path, time.time() - t_start, steps
        if node not in visited:
            visited.add(node)
            for n in G.neighbors(node):
                stack.append((n, path + [n]))
    return [], time.time() - t_start, steps

def agent_MCGS(G, start, target):

    raise NotImplemented

def agent_MCTS_UCT(G, start, target):
    #selection

    #expansion

    #simulation

    #propagation


    raise NotImplemented


def agent_MCTS_Bayes(G, start, target):
    list_visited
    # selection

    # expansion

    # simulation

    # propagation

    raise NotImplemented
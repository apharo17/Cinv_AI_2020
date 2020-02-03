import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_adj_list(df, source_col, target_col):

    nodes = set(df[source_col].tolist()) | set(df[target_col].tolist())
    n = len(nodes)
    node_to_index = {}
    i = 0
    for node in nodes:
        node_to_index[node] = i
        i += 1

    source_to_index = {}
    adj_list = []
    for index, row in df.iterrows():
        s = node_to_index[row['Source']]
        t = node_to_index[row['Target']]
        if s not in source_to_index:
            source_to_index[s] = len(adj_list)
            adj_list.append([s]) #Add head to the list
        adj_list[source_to_index[s]].append(t)

    return node_to_index, adj_list

def from_adj_list(adj_list):
    G = nx.DiGraph()
    for l in adj_list:
        s = l[0]
        for t in l[1:]:
            G.add_edge(s, t)
    return G

def multiply(head_to_index, adj_list, vec):
    """Performs the operation transpose(A)x, where A is represented by adj_list and x by vec"""
    res = np.zeros(vec.shape)
    for head in range(len(vec)):
        if vec[head] > 0:
            idx = head_to_index[head]
            for tail in adj_list[idx][1:]:
                res[tail] += 1
    return res

def bfs_matrix(n, adj_list, s, t):

    i = 0
    head_to_index = {}
    for l in adj_list:
        head = l[0]
        head_to_index[head] = i
        i += 1

    distance = np.zeros()
    distance[s] = 1
    frontier = distance

    for i in range(n):
        #Calculate the new frontier: the neighbors of the current frontier
        frontier = multiply(head_to_index, adj_list, frontier)
        #Discard what has been visited
        frontier = np.array(np.logical_and(frontier, np.logical_not(distance)), dtype='int32')

        indexes = (np.nonzero(frontier))[0] #Index 0 to discard the tuple result of np.nonzero function
        if len(indexes) == 0:
            break

        distance[indexes] += np.ones(len(indexes))


node_to_index = None
node_to_index, adj_list = get_adj_list(df, 'Source', 'Target')
G2 = from_adj_list(adj_list)
pos = nx.nx_pydot.graphviz_layout(G2)
nx.draw(G2, node_size=50, width=0.5, arrowsize=5, pos=pos)
plt.show()
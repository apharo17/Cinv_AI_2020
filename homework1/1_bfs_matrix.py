import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_graph_df(file_names):
    li = []

    for f in file_names:
        tmp = pd.read_csv(f)
        li.append(tmp)

    df = pd.concat(li, axis=0, ignore_index=True)
    return df


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


def multiply(head_to_index, adj_list, vec):
    """Performs the operation transpose(A)x, where A is represented by adj_list and x by vec"""
    res = np.zeros(vec.shape)
    for head in range(len(vec)):
        if vec[head] > 0:
            if head in head_to_index:
                idx = head_to_index[head]
                for tail in adj_list[idx][1:]:
                    res[tail] += 1
    return res


def bfs_matrix(n, adj_list, s, t, verbose=False):

    i = 0
    head_to_index = {}
    for l in adj_list:
        head = l[0]
        head_to_index[head] = i
        i += 1

    visited = np.zeros(n)
    visited[s] = 1
    frontier = visited

    for i in range(n):
        #Calculate the new frontier: the neighbors of the current frontier
        frontier = multiply(head_to_index, adj_list, frontier)
        #Discard what has been visited
        frontier = np.array(np.logical_and(frontier, np.logical_not(visited)), dtype='int32')

        indexes = (np.nonzero(frontier))[0] #Index 0 to discard the tuple result of np.nonzero function

        if verbose:
            print(indexes)

        if len(indexes) == 0:
            break

        visited[indexes] = np.ones(len(indexes))
        if visited[t] > 0:
            return True

    return False


df = load_graph_df(["stormofswords.csv"])
df = df[['Source', 'Target']] #Discard weights for BFS
df.drop_duplicates(subset=['Source', 'Target'], inplace=True)

node_to_index, adj_list = get_adj_list(df, 'Source', 'Target')
s = node_to_index['Cersei']
t = node_to_index['Melisandre']
print(bfs_matrix(len(node_to_index), adj_list, s, t, verbose=True))
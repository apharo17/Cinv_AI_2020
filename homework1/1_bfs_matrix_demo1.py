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


def get_adj_matrix(df, source_col, target_col):

    nodes = set(df[source_col].tolist()) | set(df[target_col].tolist())
    n = len(nodes)
    node_to_index = {}
    i = 0
    for node in nodes:
        node_to_index[node] = i
        i += 1

    A = np.zeros((n, n))

    for index, row in df.iterrows():
        s = node_to_index[row['Source']]
        t = node_to_index[row['Target']]
        A[s][t] = 1

    #print(node_to_index)
    return node_to_index, A


def bfs(A, s, plot_step=False):
    n = A.shape[0]
    explored = set()
    frontier = list()
    frontier.append(s)

    while len(frontier) > 0:
        print(frontier)
        u = frontier.pop(0)
        if u not in explored:
            explored.add(u)
            for v in range(n):
                if A[u][v] > 0:
                    if v not in explored:
                        frontier.append(v)



df = load_graph_df(["stormofswords.csv"])
df = df[['Source', 'Target']] #Discard weights for BFS
df.drop_duplicates(subset=['Source', 'Target'], inplace=True)

node_to_index, A = get_adj_matrix(df, 'Source', 'Target')
index_to_node = {node_to_index[node]:node for node in node_to_index.keys()}
print(index_to_node)

bfs(A, 5, plot_step=False)

#G = nx.from_pandas_edgelist(df, source='Source', target='Target')
#nx.draw(G, node_size=50, width=0.5)
#plt.show()
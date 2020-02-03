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


def get_adj_list(df, source_col, target_col, node_to_index):

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
            adj_list.append([])
        adj_list[source_to_index[s]].append(t)

    return source_to_index, adj_list

def from_adj_list(source_to_index, adj_list):
    G = nx.DiGraph()
    for s in source_to_index:
        i = source_to_index[s]
        for t in adj_list[i]:
            G.add_edge(s, t)
    return G


def show_graph(s, t, explored, frontier):

    if G is not None:

        frontier_set = set(frontier)
        #print(frontier_set)
        color_map = ['blue' for id_node in G]
        i = 0
        for id_node in G:
            if id_node in explored:
                color_map[i] = 'yellow'
            elif id_node in frontier_set:
                color_map[i] = 'red'

            if id_node==s:
                color_map[i] = 'gray'
            if id_node==t:
                color_map[i] = 'black'
            #TODO:
            #s y t de negro y marcarlas con un label
            #Marcar u

            i += 1

        nx.draw(G, node_size=50, width=0.5, node_color=color_map, arrowsize=5, pos=pos)
        plt.show()


def bfs(A, s, t, plot_step=False):
    #TODO: verbose parameter

    n = A.shape[0]
    explored = set()
    frontier = []
    frontier.append(s)
    i=0

    if plot_step:
        print('-----Start-----')
        print('Frontier:', frontier)
        print('Explored:', explored)


    while len(frontier) > 0:
        i +=1

        u = frontier.pop(0)
        if u not in explored:
            explored.add(u)

            if plot_step:
                print('-----Iteration-----:', i)
                print('Frontier:', frontier)
                show_graph(s, t, explored, frontier)

            if u == t:
                return True

            for v in range(n):
                if A[u][v] > 0:
                    if v not in explored:
                        frontier.append(v)

            if plot_step:
                print('Explored:', explored)
                show_graph(s, t, explored, frontier)

    return False



df = load_graph_df(["stormofswords.csv"])
df = df[['Source', 'Target']] #Discard weights for BFS
df.drop_duplicates(subset=['Source', 'Target'], inplace=True)

#Option 1: Load from adj matrix
node_to_index, A = get_adj_matrix(df, 'Source', 'Target')
index_to_node = {node_to_index[node]:node for node in node_to_index.keys()}
#print(index_to_node)

rows, cols = np.where(A == 1)
edges = zip(rows.tolist(), cols.tolist())
G = nx.DiGraph()
G.add_edges_from(edges)
pos = nx.nx_pydot.graphviz_layout(G)

#s = node_to_index['Hodor']
#t = node_to_index['Jon']
#print(bfs(A, s, t, plot_step=True))


#Option 2: Load from adj list
node_to_index = None
source_to_index, adj_list = get_adj_list(df, 'Source', 'Target', node_to_index)
G2 = from_adj_list(source_to_index, adj_list)
pos = nx.nx_pydot.graphviz_layout(G2)
nx.draw(G2, node_size=50, width=0.5,  arrowsize=5, pos=pos)
plt.show()



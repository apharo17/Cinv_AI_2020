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

def mat2adjlist(A):
    adj_list = []
    head_to_index = {}
    for row in range(A.shape[0]):
        for col in range(A.shape[0]):
            if A[row][col] > 0:
                if row not in head_to_index:
                    head_to_index[row] = len(adj_list)
                    adj_list.append([row])
                adj_list[head_to_index[row]].append(col)
    return adj_list

def bfs_matrix(n, adj_list, s):

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


'''
#Option 1: Load from adj matrix
node_to_index, A = get_adj_matrix(df, 'Source', 'Target')
index_to_node = {node_to_index[node]:node for node in node_to_index.keys()}
#print(index_to_node)

rows, cols = np.where(A == 1)
edges = zip(rows.tolist(), cols.tolist())
G = nx.DiGraph()
G.add_edges_from(edges)
pos = nx.nx_pydot.graphviz_layout(G)

s = node_to_index['Hodor']
t = node_to_index['Jon']
print(bfs(A, s, t, plot_step=True))
'''

#Option 2: Load from adj list
node_to_index = None
node_to_index, adj_list = get_adj_list(df, 'Source', 'Target')
s = node_to_index['Cersei']
t = node_to_index['Melisandre']
print(bfs_matrix(len(node_to_index), adj_list, s, t, verbose=True))


#G2 = from_adj_list(adj_list)
#pos = nx.nx_pydot.graphviz_layout(G2)
#nx.draw(G2, node_size=50, width=0.5,  arrowsize=5, pos=pos)
#plt.show()


#Prueba de la funcion de multiplicacion
'''
A = np.zeros((7,7))

A[0][1] = A[0][2] = A[0][3] = 1
A[1][3] = A[1][4] = 1
A[2][5] = 1
A[3][2] = A[3][5] = A[3][6] = 1
A[4][3] = A[4][6] = 1
A[6][5] = 1
print(A)

adj_list = mat2adjlist(A)

vec = np.zeros(7)
vec[1] = 1
vec[2] = 1
vec[3] = 1

i = 0
head_to_index = {}
for l in adj_list:
    head = l[0]
    head_to_index[head] = i
    i += 1

print()
print(multiply(head_to_index, adj_list, vec))
'''


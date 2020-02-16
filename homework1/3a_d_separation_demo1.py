import numpy as np


def fromfile(filename):

    """Load graph from file"""
    f = open(filename)
    line = f.readline()
    n = int(line.strip())
    G = np.zeros((n,n))

    for line in f.readlines():
        args = line.split(',')
        u = int(args[0])
        v = int(args[1])
        G[u][v] = 1

    f.close()
    return G



def check_edge(G, u, v):

    """Check if there is an edge between u and v"""
    if G[u][v] == 1 or G[v][u] == 1:
        return True
    return False



def get_und_neighbors(G, u):

    """Return the neighbors of u"""
    indexes_row = set((np.nonzero(G[u, :]))[0])
    indexes_col = set((np.nonzero(G[:, u]))[0])
    return indexes_row | indexes_col



def is_legal(G, A, triple):

    """Check if the triple is legal"""
    u = triple[0]
    v = triple[1]
    w = triple[2]

    #It is not head to head and v is not in A
    if (G[v][u] > 0 and G[v][w] > 0) or (G[u][v] > 0 and G[v][w] > 0) or (G[w][v] > 0 and G[v][u] > 0):
        if v not in A:
            return True
    #It is head to head and v has a descendent in A
    if (G[u][v] > 0 and G[w][v] > 0) and descendents[v]:
        return True

    return False



def find_reachable_nodes(G, A, B):

    R = set()
    n = G.shape[0]
    visited = np.array(np.zeros((n, n)), dtype=bool)
    edge_list = []

    for u in B:
        R.add(u)
        neighbors = get_und_neighbors(G, u)
        for v in neighbors:
            R.add(v)
            edge_list.append((u,v))
            visited[u][v] = True

    found = True
    while found:

        found = False
        num_edges = len(edge_list)
        for _ in range(num_edges):
            (u, v) = edge_list.pop(0)
            neighbors = get_und_neighbors(G, v)
            for w in neighbors:
                if not visited[v][w]:
                    if is_legal(G, A, (u,v,w)):
                        R.add(w)
                        edge_list.append((v, w))
                        visited[v][w] = True
                        found = True
    return R



def check_descendents(G, s, A):

    n = G.shape[0]
    visited = np.zeros(n, dtype = bool)
    visited[s] = True
    frontier = np.zeros(n)
    frontier[s] = 1

    for i in range(n):
        #Calculate the new frontier: the neighbors of the current frontier
        frontier = np.transpose(G).dot(frontier)
        #Discard what has been visited
        frontier = np.array(np.logical_and(frontier, np.logical_not(visited)), dtype='int32')

        indexes = (np.nonzero(frontier))[0] #Index 0 to discard the tuple result of np.nonzero function
        if len(indexes) == 0:
            break

        for idx in indexes:
            if idx in A:
                return True

        visited[indexes] = np.ones(len(indexes), dtype = bool)

    return False



def find_d_separations(G, A, B):

    n = G.shape[0]
    for v in range(n):
        if v in A or check_descendents(G, v, A):
            descendents[v] = True

    R = find_reachable_nodes(G,A,B)
    return set(range(n))-(A|R)



G = fromfile('3a_data1.txt')
n = G.shape[0]
descendents = [False for i in range(n)]

###Uncomment one of the following cases###

#Case 1
#res = find_d_separations(G, set(), {0})
#Case 2
#res = find_d_separations(G, {1,6}, {0})
#Case 3
#res = find_d_separations(G, {1,4,8}, {0})
#Case 4
#res = find_d_separations(G, {1,4}, {0})
#Case 5
#res = find_d_separations(G, {2,8}, {0})
#Case 6
#res = find_d_separations(G, {1,2}, {0})
#Case 7
res = find_d_separations(G, {4,5}, {0})


print(res)

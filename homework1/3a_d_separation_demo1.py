import numpy as np

def fromfile(filename):

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


def find_d_separations(G, A, B, D):

    n = G.shape[0]
    for v in range(n):
        if v in A or check_descendents(G, v, A):
            descendents[v] = True




G = fromfile('3a_data1.txt')
n = G.shape[0]
descendents = [False for i in range(n)]

find_d_separations(G, {0}, {}, {})
print(descendents)
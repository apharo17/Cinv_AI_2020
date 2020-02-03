import numpy as np

def multiply_canon(head_to_index, adj_list, e):

    head = e
    n = len(head_to_index)
    res = np.zeros(n)

    if head in head_to_index:
        idx = head_to_index[head]
        for tail in adj_list[idx][1:]: #Skip head
            res[tail] = 1
    return res


def dfs_matrix(n, adj_list, s, t, verbose=False):

    i = 0
    head_to_index = {}
    for l in adj_list:
        head = l[0]
        head_to_index[head] = i
        i += 1

    depth = 0
    depth_to_indexes = [[] for i in n]
    visited = np.zeros(n)
    visited[s] = 1

    neighbors = np.array((n,n)) #Checar dimension
    neighbors[:, depth] = multiply_canon(head_to_index, adj_list, s) #Get neighbors of s, with depth zero
    depth_to_indexes[depth] = list(
        (np.nonzero(neighbors[:, depth]))[0])  # Index 0 to discard the tuple result of np.nonzero function

    while len(depth_to_indexes[0]) > 0:
        u = depth_to_indexes[depth].pop(0)
        depth += 1

        if u == t:
            return True

        if depth < n:

            neighbors[:, depth + 1] = multiply_canon(head_to_index, adj_list, u)
            depth_to_indexes[depth+1] = list(
                (np.nonzero(neighbors[:, depth+1]))[0])  # Index 0 to discard the tuple result of np.nonzero function

        else:
            depth -= 1

    return False


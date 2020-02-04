import numpy as np

def multiply_canon(n, head_to_index, adj_list, e):

    head = e
    res = np.zeros(n)

    if head in head_to_index:
        idx = head_to_index[head]
        for tail in adj_list[idx][1:]: #Skip head
            res[tail] = 1
    return res



def dfs_matrix(n, adj_list, s, t, labels, verbose=False):

    i = 0
    head_to_index = {}
    for l in adj_list:
        head = l[0]
        head_to_index[head] = i
        i += 1

    depth = 0
    visited = np.zeros(n)
    neighbors = np.zeros((n,n))
    depth_to_indexes = [[] for i in range(n)]

    u = s
    visited[s] = 1

    # Get neighbors of s discarding visited nodes
    temp = multiply_canon(n, head_to_index, adj_list, u)
    neighbors[:, depth] = np.array(np.logical_and(temp, np.logical_not(visited)), dtype='int32')
    depth_to_indexes[depth] = list(
        (np.nonzero(neighbors[:, depth]))[0])  # Index 0 to discard the tuple result of np.nonzero function

    while True:
        if depth == 0 and len(depth_to_indexes[depth]) == 0:
            return False

        if verbose:
            print('Depth:', depth, '\t Selected node:', labels[u])
            indexes = depth_to_indexes[depth]
            if len(indexes) > 0:
                for idx in indexes[:-1]:
                    print(labels[idx], end=', ')
                print(labels[indexes[-1]])
            else:
                print('Empty')

        if len(depth_to_indexes[depth]) > 0:
            u = depth_to_indexes[depth].pop(0)
            visited[u] = 1
            depth += 1
            if u == t:
                return True

            # Get neighbors of u discarding visited nodes
            temp = multiply_canon(n, head_to_index, adj_list, u)
            neighbors[:, depth] = np.array(np.logical_and(temp, np.logical_not(visited)), dtype='int32')
            depth_to_indexes[depth] = list(
                (np.nonzero(neighbors[:, depth]))[0])  # Index 0 to discard the tuple result of np.nonzero function

        else:
            depth -= 1

    return False


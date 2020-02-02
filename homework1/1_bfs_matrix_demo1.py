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
    nodes_dict = {}
    i = 0
    for node in nodes:
        nodes_dict[node] = i
        i += 1

    A = np.zeros((n, n))

    for index, row in df.iterrows():
        s = nodes_dict[row['Source']]
        t = nodes_dict[row['Target']]
        A[s][t] = 1

    #print(nodes_dict)
    return A


df = load_graph_df(["stormofswords.csv"])
df = df[['Source', 'Target']] #Discard weights for BFS
df.drop_duplicates(subset=['Source', 'Target'], inplace=True)

A = get_adj_matrix(df, 'Source', 'Target')

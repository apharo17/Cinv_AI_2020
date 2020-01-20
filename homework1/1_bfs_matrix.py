import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

all_books = ["stormofswords.csv"]

li = []

for f in all_books:
    tmp = pd.read_csv(f)
    li.append(tmp)

df = pd.concat(li, axis=0, ignore_index=True)

df = df[['Source', 'Target']]
df.drop_duplicates(subset=['Source', 'Target'], inplace=True)

# create the networkx object

G = nx.from_pandas_edgelist(df, source='Source', target='Target')

nx.draw(G, node_size=50, width=0.5)
plt.show()
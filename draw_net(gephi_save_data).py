import random
import networkx as nx
import pandas as pd

N = 1000
time = 1000
G = nx.barabasi_albert_graph(N, 2)

edges = []
id = [i for i in G.nodes()]
label = id
interval = ["" for i in range(N)]
score = ["" for i in range(N)]


for t in range(time):
    for node in range(N):
        state = random.randint(0, 2)  # S:0 R:1 V:2
        if state != 2:
            if len(interval[node]) == 0:
                interval[node] += "<[{},{}]".format(t,t+1)
            else:
                interval[node] += ";[{},{}]".format(t,t+1)
        if len(score[node]) == 0:
            score[node] += "<[{},{},{}]".format(t,t+1,state)
        else:
            score[node] += ";[{},{},{}]".format(t,t+1,state)

for i in range(len(interval)):
    if len(interval[i]) != 0:
        interval[i] += '>'
for i in range(len(score)):
    if len(score[i]) != 0:
        score[i] += '>'

df1 = pd.DataFrame({"id":id, "label":label, "interval":interval, "score":score})
# print(df1)

for source, target in G.edges():
    # print(source, target)
    edges.append([source, target])
df2 = pd.DataFrame(edges, columns=['source', 'target'])

with pd.ExcelWriter("excel_writer.xlsx") as writer:
    df1.to_excel(excel_writer=writer,sheet_name="node",index=None)
    df2.to_excel(excel_writer=writer,sheet_name="egde",index=None)



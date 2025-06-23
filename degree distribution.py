import networkx as nx
import matplotlib.pyplot as plt

# 创建一个随机图
G = nx.barabasi_albert_graph(1000, 2)

# 计算每个节点的度
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

print(degree_sequence[:10], sum(degree_sequence[:10])/4000)
print(degree_sequence[:20], sum(degree_sequence[:20])/4000)

# 绘制度分布图
plt.figure(figsize=(10, 6))
plt.hist(degree_sequence, bins=range(max(degree_sequence) + 1), alpha=0.7, color='blue', edgecolor='black')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")
plt.grid(True)
plt.show()

# 绘制度分布图
plt.figure(figsize=(10, 6))
plt.hist(degree_sequence[:20], bins=range(max(degree_sequence) + 1), alpha=0.7, color='blue', edgecolor='black')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")
plt.grid(True)
plt.show()

# 绘制度分布图
plt.figure(figsize=(10, 6))
plt.hist(degree_sequence[:50], bins=range(max(degree_sequence) + 1), alpha=0.7, color='blue', edgecolor='black')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")
plt.grid(True)
plt.show()
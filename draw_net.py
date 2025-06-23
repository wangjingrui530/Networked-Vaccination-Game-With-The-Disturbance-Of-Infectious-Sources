import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle

def SaveParameters(paras):
    with open('obj.pickle', 'wb') as f:
            pickle.dump(paras, f)

# 创建一个有1000个节点的随机图
G = nx.erdos_renyi_graph(1000, 0.001)  # 参数分别是节点数和概率
# 绘制网络图
plt.figure(dpi=300)
# plt.figure(dpi=1000, figsize=(12, 12))  # 设置画布大小以适应大量节点
# pos = nx.circular_layout(G)  # 使用圆形布局
pos = nx.spring_layout(G)
print(pos)
print(type(pos))
nx.draw(G, pos, with_labels=False, node_size=10, edge_color='gray', node_color='lightblue')
plt.axis('off')  # 关闭坐标轴
# plt.show()
plt.savefig('test.jpg')

SaveParameters([G, pos])
# 从pickle文件中读取Python对象
with open('obj.pickle', 'rb') as f:
    loaded_obj = pickle.load(f)
print(loaded_obj)




# # 创建一个图
# G = nx.barabasi_albert_graph(200, 2, seed=42)  # 使用 Barabási-Albert 模型生成无标度网络
#
# # 计算节点属性
# node_degrees = dict(G.degree())  # 节点度数
# max_degree = max(node_degrees.values())  # 最大度数
#
# # 节点大小基于度数
# node_sizes = [degree * 50 for degree in node_degrees.values()]  # 度数越大，节点越大
#
# # 节点颜色分配
# # 假设我们将节点分为三类，分别用蓝色、红色和绿色表示
# # 这里我们随机分配颜色，你可以根据实际属性调整
# np.random.seed(42)  # 固定随机种子，保证结果可复现
# node_colors = np.random.choice(['blue', 'red', 'green'], size=len(G.nodes()))
#
# # 使用 Kamada-Kawai 布局
# pos = nx.kamada_kawai_layout(G)
#
# # 绘制图形
# plt.figure(figsize=(12, 10), dpi=300)
#
# # 绘制节点
# nx.draw_networkx_nodes(
#     G, pos,
#     node_size=node_sizes,
#     node_color=node_colors,
#     alpha=0.8  # 节点透明度
# )
#
# # 绘制边
# nx.draw_networkx_edges(
#     G, pos,
#     width=0.5,  # 边的宽度
#     alpha=0.3,  # 边的透明度
#     edge_color="gray"  # 边的颜色
# )
#
# # 添加颜色条（可选）
# from matplotlib.lines import Line2D
# from matplotlib import cm
#
# # 创建颜色条
# colors = ['blue', 'red', 'green']
# labels = ['Category 1', 'Category 2', 'Category 3']
# custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
#
# plt.legend(custom_lines, labels, title="Node Categories")
#
# # 美化细节
# plt.title("Advanced Network Visualization with Custom Colors", fontsize=16, fontweight="bold")
# plt.axis("off")  # 关闭坐标轴
# plt.show()
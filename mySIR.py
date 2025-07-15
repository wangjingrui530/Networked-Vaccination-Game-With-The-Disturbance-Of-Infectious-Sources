import math
import random
import networkx as nx
import numpy as np
from numba import jit
import IM


def Lattice(L):
    G = nx.Graph()
    G.add_nodes_from(list(range(L * L)))
    for i in range(L):
        for j in range(L):
            G.add_edge((i * L + j), (j + 1) % L + i * L)
    for j in range(L):
        for i in range(L):
            G.add_edge(i * L + j, ((i + 1) % L) * L + j)
    return G


def Graph_generation(size, average_degree, net_type="complete", seed = None):
    if net_type == 'ba':
        if seed == None:
            G = nx.barabasi_albert_graph(size, int(average_degree / 2))
        else:
            G = nx.barabasi_albert_graph(size, int(average_degree / 2), seed=seed)
    if net_type == 'ws':
        if seed == None:
            G = nx.watts_strogatz_graph(size, int(average_degree / 2), p=0.1)
        else:
            G = nx.watts_strogatz_graph(size, int(average_degree / 2), p=0.1, seed=seed)
    if net_type == 'er':
        if seed == None:
            G = nx.random_graphs.erdos_renyi_graph(size, average_degree / (size - 1))
        else:
            G = nx.random_graphs.erdos_renyi_graph(size, average_degree / (size - 1), seed=seed)
    if net_type == 'la':
        G = Lattice(int(math.sqrt(size)))
    return G


def GraphToArray(G):
    neighborsList = []
    neighborsArray = []
    for i in G.nodes():  # initialize neighbors
        neighbors = list(G.adj[i])
        neighborsList.append(neighbors)
        temp = np.hstack((neighbors, np.linspace(-1, -1, 1000 - len(neighbors)))).tolist()
        neighborsArray.append(temp)
    neighborsArray = np.asarray(neighborsArray, dtype='int32')
    return neighborsList, neighborsArray


@jit(nopython=True)
def NumOfVaccination(total_attribute):
    num = 0
    for i in total_attribute[1]:
        if i == 0:
            num += 1
    return num


@jit(nopython=True)
def InitalizationVac(total_attribute, size):
    for i in range(size):
        if random.random() <= 0.5:
            total_attribute[0][i] = 0   # vac
            total_attribute[1][i] = 0
        else:
            total_attribute[0][i] = 1   # non-vac
            total_attribute[1][i] = 1


def ChooseInfectSeeds(total_attribute, G, size, infectseeds, attack):
    seeds = np.zeros(infectseeds)

    if attack == "None":
        return seeds

    vacnum = 0
    for i in total_attribute[1]:
        if i == 0:
            vacnum += 1
    if (size - vacnum) <= infectseeds:
        return seeds

    if attack == "Random":
        return IM.Random(total_attribute, seeds, infectseeds, size)

    if attack == "HDA":
        return IM.DegreeCentrality(total_attribute, seeds, infectseeds, G)

    if attack == "induct":
        return IM.induct(total_attribute, seeds, infectseeds, G)

    if attack == "PG":
        return IM.PageRank(total_attribute, seeds, infectseeds, G)

    if attack == "KS":
        return IM.KShell(total_attribute, seeds, infectseeds, G)

    if attack == "EC":
        return IM.EigenvectorCentrality(total_attribute, seeds, infectseeds, G)

    if attack == "CC":
        return IM.ClosenessCentrality(total_attribute, seeds, infectseeds, G)

    if attack == "BC":
        return IM.BetweennessCentrality(total_attribute, seeds, infectseeds, G)
    
    if attack == "DR":
        return IM.DomiRank(total_attribute, seeds, infectseeds, G)
    
    if attack == "CI":
        return IM.CollectiveInfluence(total_attribute, seeds, infectseeds, G)


@jit(nopython=True)
def InitalizationInfect(total_attribute, infectseeds, seeds, size):
    for i in range(size):
        total_attribute[2][int(i)] = 0

    if sum(seeds) == 0:
        return

    vacnum = 0
    for i in total_attribute[1]:
        if i == 0:
            vacnum += 1
    if (size - vacnum) <= infectseeds:
        return

    for i in seeds:
        total_attribute[2][int(i)] = 1


@jit(nopython=True)
def TransitionRateSum(total_attribute, neighborsArray, size, r, g):
    lamda = 0
    for i in range(size):
        if total_attribute[1][i] == 0:
            total_attribute[4][i] = 0
            continue
        else:
            if total_attribute[2][i] == 0:
                num = 0
                for j in neighborsArray[i]:
                    if j == -1:
                        break
                    if total_attribute[2][j] == 1:
                        num += 1
                total_attribute[4][i] = r * num
                lamda += total_attribute[4][i]
                continue
            if total_attribute[2][i] == 1:
                total_attribute[4][i] = g
                lamda += g
                continue
    return lamda


@jit(nopython=True)
def GillespieAlgorithm(total_attribute, neighborsArray, lamda, infectseeds, size, r, g):
    I = infectseeds
    if lamda == 0:
        return 0
    infect_nums = I

    while I > 0:
        v = random.random()
        temp = 0
        for i in range(size):
            temp += total_attribute[4][i]
            if lamda != 0:
                if temp / lamda > v:
                    agent = i
                    break

        if total_attribute[2][agent] == 0 and total_attribute[1][agent] == 1:
            total_attribute[2][agent] = 1
            I += 1
            infect_nums += 1
            lamda = lamda + (g - total_attribute[4][agent])
            total_attribute[4][agent] = g
            for j in neighborsArray[agent]:
                if j == -1:
                    break
                if total_attribute[2][j] == 0 and total_attribute[1][j] == 1:
                    total_attribute[4][j] += r
                    lamda += r
            continue

        if total_attribute[2][agent] == 1:
            total_attribute[2][agent] = 2
            I -= 1
            lamda -= g
            total_attribute[4][agent] = 0
            for j in neighborsArray[agent]:
                if j == -1:
                    break
                if total_attribute[2][j] == 0 and total_attribute[1][j] == 1:
                    total_attribute[4][j] -= r
                    lamda -= r
            continue
    return infect_nums


@jit(nopython=True)
def CalCost(total_attribute, size, c):
    for i in range(size):
        if total_attribute[1][i] == 0:
            total_attribute[3][i] = -c
            # print(total_attribute[3][i])
        else:
            if total_attribute[2][i] != 0:
                total_attribute[3][i] = -1
                # print(total_attribute[3][i])
            else:
                total_attribute[3][i] = 0
    return


@jit(nopython=True)
def FeimiFunction(total_attribute, i, j, K):
    costi = total_attribute[3][i]
    costj = total_attribute[3][j]
    prob = 1 / (1 + math.exp((costi - costj) / K))
    return prob


@jit(nopython=True)
def Imitation(total_attribute, neighborsArray, size, K):
    for i in range(size):
        neighbors_number = 0
        for j in neighborsArray[i]:
            if j == -1:
                break
            neighbors_number += 1
        if neighbors_number == 0:
            continue
        imitation_node = neighborsArray[i][int(random.randrange(neighbors_number))]
        prob = FeimiFunction(total_attribute, i, imitation_node, K)
        if prob >= random.random():
            total_attribute[0][i] = total_attribute[1][imitation_node]
        else:
            total_attribute[0][i] = total_attribute[1][i]


@jit(nopython=True)
def UpdateStrategy(total_attribute, size):
    for i in range(size):
        total_attribute[1][i] = total_attribute[0][i]


@jit(nopython=True)
def UpperBound(total_attribute, size):
    infect_nums = 0
    for size_ in range(size):
        if total_attribute[1][size_] == 1:
            total_attribute[2][size_] = 2
            infect_nums += 1
        else:
            total_attribute[2][size_] = 0
    return infect_nums


@jit(nopython=True)
def LowerBound(total_attribute, neighborsArray, size):
    infect_nums = 0
    for size_ in range(size):
        if total_attribute[1][size_] == 1:
            total_attribute[2][size_] = 2
            infect_nums += 1
            for neig in neighborsArray[size_]:
                if neig == -1:
                    break
                if total_attribute[1][neig] == 0:
                    total_attribute[2][size_] = 0
                    infect_nums -= 1
                    break
        else:
            total_attribute[2][size_] = 0

    return infect_nums
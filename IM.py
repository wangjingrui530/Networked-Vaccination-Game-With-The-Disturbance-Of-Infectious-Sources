import math
import random
import networkx as nx
import numpy as np
from numba import jit


def GetVacNodes(total_attribute, compare = False):
    if compare:
        vacnodes = []
        for i in range(len(total_attribute[0])):
            if total_attribute[0][i] == 0:
                vacnodes.append(i)
        return vacnodes
    else:
        vacnodes = []
        for i in range(len(total_attribute[1])):
            if total_attribute[1][i] == 0:
                vacnodes.append(i)
        return vacnodes


@jit(nopython=True)
def Random(total_attribute, seeds, infectseeds, size):
    seeds_num = 0
    while seeds_num < infectseeds:
        i = random.randrange(size)
        if total_attribute[1][int(i)] != 0:
            not_in = 1
            for j in seeds[0:seeds_num]:
                if i == j:
                    not_in = 0
            if not_in:
                seeds[seeds_num] = i
                seeds_num += 1
    return seeds


def DegreeCentrality(total_attribute, seeds, infectseeds, G):
    remove_G = G.copy()
    vacnodes = GetVacNodes(total_attribute)
    remove_G.remove_nodes_from(vacnodes)

    sorted_d = dict(remove_G.degree())
    sorted_d = sorted(sorted_d.items(), key=lambda x: x[1], reverse=True)

    for node, value in sorted_d:
        seeds[infectseeds - 1] = node
        infectseeds -= 1
        if infectseeds == 0:
            return seeds


def induct(total_attribute, seeds, infectseeds, G):
    remove_G = G.copy()
    vacnodes = GetVacNodes(total_attribute)
    remove_G.remove_nodes_from(vacnodes)

    ccs = [cc for cc in nx.connected_components(remove_G)]
    ccs = sorted(ccs, key=lambda x: len(x), reverse=False)

    cc_set = []
    index = 0
    for cc in ccs:
        for node in cc:
            seeds[infectseeds-1] = node
            infectseeds -= 1
            cc_set.append(index)
            if infectseeds == 0:
                return seeds
        index += 1


def PageRank(total_attribute, seeds, infectseeds, G):
    remove_G = G.copy()
    vacnodes = GetVacNodes(total_attribute)
    remove_G.remove_nodes_from(vacnodes)

    sorted_pg = nx.pagerank(remove_G)
    sorted_pg = sorted(sorted_pg.items(), key=lambda x: x[1], reverse=True)

    for node, value in sorted_pg:
        seeds[infectseeds - 1] = node
        infectseeds -= 1
        if infectseeds == 0:
            return seeds


def KShell(total_attribute, seeds, infectseeds, G):
    remove_G = G.copy()
    vacnodes = GetVacNodes(total_attribute)
    remove_G.remove_nodes_from(vacnodes)

    sorted_ks = nx.core_number(remove_G)
    sorted_ks = sorted(sorted_ks.items(), key=lambda x: x[1], reverse=True)

    for node, value in sorted_ks:
        seeds[infectseeds - 1] = node
        infectseeds -= 1
        if infectseeds == 0:
            return seeds


def EigenvectorCentrality(total_attribute, seeds, infectseeds, G):
    remove_G = G.copy()
    vacnodes = GetVacNodes(total_attribute)
    remove_G.remove_nodes_from(vacnodes)

    sorted_ec = nx.eigenvector_centrality(remove_G, max_iter=500, tol=1.0e-3)
    sorted_ec = sorted(sorted_ec.items(), key=lambda x: x[1], reverse=True)

    for node, value in sorted_ec:
        seeds[infectseeds - 1] = node
        infectseeds -= 1
        if infectseeds == 0:
            return seeds


def ClosenessCentrality(total_attribute, seeds, infectseeds, G):
    remove_G = G.copy()
    vacnodes = GetVacNodes(total_attribute)
    remove_G.remove_nodes_from(vacnodes)

    sorted_cc = nx.closeness_centrality(remove_G)
    sorted_cc = sorted(sorted_cc.items(), key=lambda x: x[1], reverse=True)

    for node, value in sorted_cc:
        seeds[infectseeds - 1] = node
        infectseeds -= 1
        if infectseeds == 0:
            return seeds


def BetweennessCentrality(total_attribute, seeds, infectseeds, G):
    remove_G = G.copy()
    vacnodes = GetVacNodes(total_attribute)
    remove_G.remove_nodes_from(vacnodes)

    sorted_bc = nx.betweenness_centrality(remove_G)
    sorted_bc = sorted(sorted_bc.items(), key=lambda x: x[1], reverse=True)

    for node, value in sorted_bc:
        seeds[infectseeds - 1] = node
        infectseeds -= 1
        if infectseeds == 0:
            return seeds
        
        
def DomiRank(total_attribute, seeds, infectseeds, G):
    remove_G = G.copy()
    vacnodes = GetVacNodes(total_attribute)
    remove_G.remove_nodes_from(vacnodes)

    GAdj = nx.to_scipy_sparse_array(remove_G)
    lambN = find_eigenvalue(GAdj, maxIter = 500, dt = 0.025, checkStep = 25) 
    sigma = - 0.12 / lambN

    _, ourDomiRankDistribution = domirank(GAdj, analytical = False, sigma = sigma)
    node_map = list(remove_G.nodes())
    zipped = dict(zip(node_map, ourDomiRankDistribution))
    DR = sorted(zipped, reverse = True, key = zipped.get)

    for node in DR:
        seeds[infectseeds - 1] = node
        infectseeds -= 1
        if infectseeds == 0:
            return seeds
    return seeds


def CollectiveInfluence(total_attribute, seeds, infectseeds, G):
    l = 2 # depth
    remove_G = G.copy()
    vacnodes = GetVacNodes(total_attribute)
    remove_G.remove_nodes_from(vacnodes)
    
    # get CI centrality
    CI = {}
    neigbors_in_l = {}
    for node_i in remove_G.nodes():
        node = node_i
        CI[node] = 0
        neigbors_in_l[node_i] = {}
        
        layers = dict(nx.bfs_successors(remove_G, source=node, depth_limit=l))
        nodes = [node]
        for i in range(1, l+1):
            neigbors_in_l[node_i][i] = []
            for x in nodes:
                # print(x, layers.get(x,[]))
                neigbors_in_l[node_i][i].extend(layers.get(x,[]))
            nodes = neigbors_in_l[node_i][i]

    for node_i in neigbors_in_l.keys():
        temp_sum = 0
        for i in range(1, l+1):
            if neigbors_in_l[node_i][i] != {}:
                for node_j in neigbors_in_l[node_i][i]:
                    temp_sum += (remove_G.degree(node_j) - 1)
        CI[node_i] = remove_G.degree(node_i) * temp_sum
    
    # sort
    CI = sorted(CI, reverse = True, key = CI.get) # max->min
    for node in CI:
        seeds[infectseeds - 1] = node
        infectseeds -= 1
        if infectseeds == 0:
            return seeds
    return seeds


def domirank(G, analytical = True, sigma = -1, dt = 0.1, epsilon = 1e-5, maxIter = 1000, checkStep = 10):
    '''
    G is the input graph as a (preferably) sparse array.
    This solves the dynamical equation presented in the Paper: "DomiRank Centrality: revealing structural fragility of
complex networks via node dominance" and yields the following output: bool, DomiRankCentrality
    Here, sigma needs to be chosen a priori.
    dt determines the step size, usually, 0.1 is sufficiently fine for most networks (could cause issues for networks
    with an extremely high degree, but has never failed me!)
    maxIter is the depth that you are searching with in case you don't converge or diverge before that.
    Checkstep is the amount of steps that you go before checking if you have converged or diverged.
    
    
    This algorithm scales with O(m) where m is the links in your sparse array.
    '''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        G = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph 
    else:
        G = G.copy()
    if analytical == False:
        if sigma == -1:
            sigma, _ = optimal_sigma(G, analytical = False, dt=dt, epsilon=epsilon, maxIter = maxIter, checkStep = checkStep)
        pGAdj = sigma*G.astype(np.float64)
        Psi = np.ones(pGAdj.shape[0]).astype(np.float64)/pGAdj.shape[0]
        maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float64)
        dt = np.float64(dt)
        j = 0
        boundary = epsilon*pGAdj.shape[0]*dt
        for i in range(maxIter):
            tempVal = ((pGAdj @ (1-Psi)) - Psi)*dt
            Psi += tempVal.real
            if i% checkStep == 0:
                if np.abs(tempVal).sum() < boundary:
                    break
                maxVals[j] = tempVal.max()
                if i == 0:
                    initialChange = maxVals[j]
                if j > 0:
                    if maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2]:
                        return False, Psi
                j+=1

        return True, Psi
    else:
        if sigma == -1:
            sigma = optimal_sigma(G, analytical = True, dt=dt, epsilon=epsilon, maxIter = maxIter, checkStep = checkStep)
        Psi = sp.sparse.linalg.spsolve(sigma*G + sp.sparse.identity(G.shape[0]), sigma*G.sum(axis=-1))
        return True, Psi
    
def find_eigenvalue(G, minVal = 0, maxVal = 1, maxDepth = 100, dt = 0.1, epsilon = 1e-5, maxIter = 100, checkStep = 10):
    '''
    G: is the input graph as a sparse array.
    Finds the largest negative eigenvalue of an adjacency matrix using the DomiRank algorithm.
    Currently this function is only single-threaded, as the bisection algorithm only allows for single-threaded
    exection. Note, that this algorithm is slightly different, as it uses the fact that DomiRank diverges
    at values larger than -1/lambN to its benefit, and thus, it is not exactly bisection theorem. I haven't
    tested in order to see which exact value is the fastest for execution, but that will be done soon!
    Some notes:
    Increase maxDepth for increased accuracy.
    Increase maxIter if DomiRank doesn't start diverging within 100 iterations -- i.e. increase at the expense of 
    increased computational cost if you want potential increased accuracy.
    Decrease checkstep for increased error-finding for the values of sigma that are too large, but higher compcost
    if you are frequently less than the value (but negligible compcost).
    '''
    x = (minVal + maxVal)/G.sum(axis=-1).max()
    minValStored = 0
    for i in range(maxDepth):
        if maxVal - minVal < epsilon:
            break
        if domirank(G, False, x, dt, epsilon, maxIter, checkStep)[0]:
            minVal = x
            x = (minVal + maxVal)/2
            minValStored = minVal
        else:
            maxVal = (x + maxVal)/2
            x = (minVal + maxVal)/2
        # if minVal == 0:
        #     print(f'Current Interval : [-inf, -{1/maxVal}]')
        # else:
        #     print(f'Current Interval : [-{1/minVal}, -{1/maxVal}]')
    finalVal = (maxVal + minVal)/2
    return -1/finalVal
import math
import random
import networkx as nx
import numpy as np
from numba import jit
import IM


def GetVacNodes(total_attribute):
    vacnodes = []
    for i in range(len(total_attribute[1])):
        if total_attribute[1][i] == 0:
            vacnodes.append(i)
    return vacnodes


# def GetTopNodes():
#     return


def AnalyseVacHDnode(total_attribute, G, top_n_node):
    number = 0
    degrees = G.degree()
    G_sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)
    for node, degree in G_sorted_degrees[:top_n_node]:
        if total_attribute[1][node] == 0:
            number += 1
    return number/top_n_node


def AnalyseSeedsDistribution(total_attribute, G, top_n_node, seeds):
    degrees = G.degree()
    G_sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)
    remove_G = G.copy()
    vacnodes = GetVacNodes(total_attribute)
    remove_G.remove_nodes_from(vacnodes)

    cc_hdnodes_set = []
    cc_seeds_set = []

    index = 0
    for cc in nx.connected_components(remove_G):
        for node, degree in G_sorted_degrees[:top_n_node]:
            if total_attribute[0][node] != 0:
                if node in cc:
                    cc_hdnodes_set.append(index)
        for seed in seeds:
            if seed in cc:
                cc_seeds_set.append(index)
        index += 1

    proportion = 0
    for cc_i in cc_seeds_set:
        if cc_i in cc_hdnodes_set:
            proportion += 1
    if proportion == 0:
        return 0
    else:
        return proportion/len(cc_seeds_set)



def AnalyseFR(total_attribute, neighborsArray, G, size, top_n_node):
    fr_number = 0
    fr_between_hdnodes_number = 0
    fr_between_vac_number = 0
    fr_between_vac_rate = 0

    degrees = G.degree()
    G_sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)

    for i in range(size):
        if total_attribute[1][i] == 1 and total_attribute[2][i] == 0:
            fr_number += 1

    for node, degree in G_sorted_degrees[:top_n_node]:
        for j in neighborsArray[node]:
            if j == -1:
                break
            if total_attribute[1][j] == 1 and total_attribute[2][j] == 0:
                fr_between_hdnodes_number += 1

    for i in range(size):
        if total_attribute[1][i] == 0:
            fr_between_vac_rate += G.degree(i)
            for j in neighborsArray[i]:
                if j == -1:
                    break
                if total_attribute[1][j] == 1 and total_attribute[2][j] == 0:
                    fr_between_vac_number += 1
    if fr_between_vac_rate == 0:
        fr_between_vac_rate = 0
    else:
        fr_between_vac_rate = fr_between_vac_number / fr_between_vac_rate # 这个参数需要再仔细推敲一下!!!

    return fr_number, fr_between_hdnodes_number, fr_between_vac_number, fr_between_vac_rate


def AnalyseStateTransition(total_attribute, G, size, top_n_node):
    # s(nv)_nv; r_nv; s(nv)_v; r_v; v_v; nv_nv; v_nv; nv_v;
    statetrans = [0, 0, 0, 0, 0, 0, 0, 0]
    top_nodes_statetrans = [0, 0, 0, 0, 0, 0, 0, 0]

    degrees = G.degree()
    G_sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)

    for i in range(size):
        if total_attribute[2][i] == 0 and total_attribute[1][i] == 1 and total_attribute[0][i] == 1:
            statetrans[0] += 1 / size # s(nv)_nv
        if total_attribute[2][i] == 2 and total_attribute[0][i] == 1:
            statetrans[1] += 1 / size # r_nv
        if total_attribute[2][i] == 0 and total_attribute[1][i] == 1 and total_attribute[0][i] == 0:
            statetrans[2] += 1 / size # s(nv)_v
        if total_attribute[2][i] == 2 and total_attribute[0][i] == 0:
            statetrans[3] += 1 / size # r_v

        if total_attribute[1][i] == 0 and total_attribute[0][i] == 0:
            statetrans[4] += 1 / size # v_v
        if total_attribute[1][i] == 1 and total_attribute[0][i] == 1:
            statetrans[5] += 1 / size # nv_nv
        if total_attribute[1][i] == 0 and total_attribute[0][i] == 1:
            statetrans[6] += 1 / size # v_nv
        if total_attribute[1][i] == 1 and total_attribute[0][i] == 0:
            statetrans[7] += 1 / size # nv_v


    for i, degree in G_sorted_degrees[:top_n_node]:
        if total_attribute[2][i] == 0 and total_attribute[1][i] == 1 and total_attribute[0][i] == 1:
            top_nodes_statetrans[0] += 1 / top_n_node  # s(nv)_nv
        if total_attribute[2][i] == 2 and total_attribute[0][i] == 1:
            top_nodes_statetrans[1] += 1 / top_n_node # r_nv
        if total_attribute[2][i] == 0 and total_attribute[1][i] == 1 and total_attribute[0][i] == 0:
            top_nodes_statetrans[2] += 1 / top_n_node  # s(nv)_v
        if total_attribute[2][i] == 2 and total_attribute[0][i] == 0:
            top_nodes_statetrans[3] += 1 / top_n_node  # r_v

        if total_attribute[1][i] == 0 and total_attribute[0][i] == 0:
            top_nodes_statetrans[4] += 1 / top_n_node  # v_v
        if total_attribute[1][i] == 1 and total_attribute[0][i] == 1:
            top_nodes_statetrans[5] += 1 / top_n_node  # nv_nv
        if total_attribute[1][i] == 0 and total_attribute[0][i] == 1:
            top_nodes_statetrans[6] += 1 / top_n_node  # v_nv
        if total_attribute[1][i] == 1 and total_attribute[0][i] == 0:
            top_nodes_statetrans[7] += 1 / top_n_node  # nv_v

    return statetrans, top_nodes_statetrans


def GetDynamicGraphData(total_attribute, seeds, t, size, interval, score):
    for node in range(size):
        # state V:0; S:1; R:2; seeds:3
        if total_attribute[1][node] == 0:
            state = 0
        else:
            if total_attribute[2][node] == 0:
                state = 1
            else:
                if node in seeds:
                    state = 3
                else:
                    state = 2

        if state != 0:
            if len(interval[node]) == 0:
                interval[node] += "<[{},{}]".format(t, t + 1)
            else:
                interval[node] += ";[{},{}]".format(t, t + 1)
        if len(score[node]) == 0:
            score[node] += "<[{},{},{}]".format(t, t + 1, state)
        else:
            score[node] += ";[{},{},{}]".format(t, t + 1, state)


def AnalyseConnectedComponent(total_attribute, G, seeds, infect_nums, compare = False):
    if compare:
        if sum(seeds) == 0:
            return 0
        else:
            remove_G = G.copy()
            vacnodes = IM.GetVacNodes(total_attribute, compare)
            remove_G.remove_nodes_from(vacnodes)
            return len(max(nx.connected_components(remove_G), key=lambda x: len(x)))

    remove_G = G.copy()
    vacnodes = IM.GetVacNodes(total_attribute)
    remove_G.remove_nodes_from(vacnodes)

    if sum(seeds) == 0:
        index = 0
        for cc in nx.connected_components(remove_G):
            index += 1
        return index, 0, 0, 0, 0

    with_seed_cc = {}
    index = 0
    for cc in nx.connected_components(remove_G):
        for seed in seeds:
            if seed in cc:
                if index not in with_seed_cc.keys():
                    with_seed_cc[index] = len(cc)
                    break
        index += 1

    with_seed_cc_size = sum(with_seed_cc.values())
    if with_seed_cc_size == 0:
        with_seed_cc_infect_rate = 0
    else:
        with_seed_cc_infect_rate = infect_nums / with_seed_cc_size
    if remove_G.number_of_nodes() == 0:
        with_seed_cc_allcc_size_rate = 0
    else:
        with_seed_cc_allcc_size_rate = with_seed_cc_size / remove_G.number_of_nodes()

    return index, with_seed_cc_size, with_seed_cc_infect_rate, with_seed_cc_allcc_size_rate, len(max(nx.connected_components(remove_G), key=lambda x:len(x)))


import numpy as np
import argparse
from tqdm import trange
import pandas as pd
import mySIR
import Analyse
import Plot


def run_x_time(args):
    # parameters
    size = args.size
    average_degree = args.average_degree
    net_type = args.net_type
    r = args.r
    g = args.g
    c = args.c
    infectseeds = args.infectseeds # number
    k = args.k
    attack = args.attack
    analyse = args.analyse

    # vac: vacstate=0; non-vac: vacstate=1
    # S: state=0; I: state=1; R: state=2
    # 0:tempvacstate, 1:vacstate, 2:state, 3:payoff, 4:transition_rate
    total_attribute = np.asarray([[0] * size for i in range(5)], dtype='float32')

    # results
    average = args.average
    iteration = args.iteration
    path = r"result\\" + net_type + "_" + attack + "_c{}".format(c) + "_time_k{}".format(k)
    result_vac = [0 for i in range(iteration)]
    result_infect = [0 for i in range(iteration)]
    result_S_infect = [0 for i in range(iteration)]
    result_sc = [0 for i in range(iteration)]
    result_average_sc = [0 for i in range(iteration)]
    result_total_sc = [0 for i in range(iteration)]

    # analyse result
    if analyse:
        top_n_node = 20
        result_hdnodes_vac = [0 for i in range(iteration)]
        result_seeds_distribution = [0 for i in range(iteration)]
        result_cc_number = [0 for i in range(iteration)]
        result_with_seed_cc_size = [0 for i in range(iteration)]
        result_with_seed_cc_infect_rate = [0 for i in range(iteration)]
        result_with_seed_cc_allcc_size_rate = [0 for i in range(iteration)]
        result_max_cc_size = [0 for i in range(iteration)]
        result_free_riders = [0 for i in range(iteration)]
        result_free_riders_between_hdnodes = [0 for i in range(iteration)]
        result_free_riders_between_vac = [0 for i in range(iteration)]
        result_free_riders_between_vac_rate = [0 for i in range(iteration)]
        result_states_trans = [[0 for i in range(iteration)] for j in range(8)]
        result_top_nodes_state_trans = [[0 for i in range(iteration)] for j in range(8)]

    for aver_time in range(average):
        # Initalize Graph
        G = mySIR.Graph_generation(size, average_degree, net_type)
        # G = mySIR.Graph_generation(size, average_degree, net_type, seed=1234) # !!!
        neighborsList, neighborsArray = mySIR.GraphToArray(G)
        if attack == 'CC' or attack == 'BC':
            total_attribute[1][:] = 1
            nonvac_seeds = mySIR.ChooseInfectSeeds(total_attribute, G, size, infectseeds, attack)
        mySIR.InitalizationVac(total_attribute, size)

        if analyse:
            edges = [[source, target] for source, target in G.edges()]
            id = [i for i in G.nodes()]
            label = id
            interval = ["" for i in range(size)]
            score = ["" for i in range(size)]

        for timestep in trange(iteration):
            # print('===============',timestep,'=================')
            if attack == 'CC' or attack == 'BC':
                if sum(total_attribute[1]) == size: # non-vac
                    seeds = nonvac_seeds
                else:
                    seeds = mySIR.ChooseInfectSeeds(total_attribute, G, size, infectseeds, attack)
            else:
                seeds = mySIR.ChooseInfectSeeds(total_attribute, G, size, infectseeds, attack)

            if analyse:
                result_hdnodes_vac[timestep] += Analyse.AnalyseVacHDnode(total_attribute, G, top_n_node) / average
                result_seeds_distribution[timestep] += Analyse.AnalyseSeedsDistribution(total_attribute, G, top_n_node, seeds) / average

            # SIR
            mySIR.InitalizationInfect(total_attribute, infectseeds, seeds, size)
            lamda = mySIR.TransitionRateSum(total_attribute, neighborsArray, size, r, g)
            infect_nums = mySIR.GillespieAlgorithm(total_attribute, neighborsArray, lamda, infectseeds, size, r, g)
            mySIR.CalCost(total_attribute, size, c)

            # Learning Dynamics
            result_vac[timestep] += (mySIR.NumOfVaccination(total_attribute) / size / average)
            result_infect[timestep] += (infect_nums / size / average)
            if infect_nums != 0:
                result_S_infect[timestep] += (infect_nums / (size - mySIR.NumOfVaccination(total_attribute))) / average

            mySIR.Imitation(total_attribute, neighborsArray, size, k)

            if analyse:
                temp_fr = Analyse.AnalyseFR(total_attribute, neighborsArray, G, size, top_n_node)
                temp_state = Analyse.AnalyseStateTransition(total_attribute, G, size, top_n_node)
                result_free_riders[timestep] += temp_fr[0] / average
                result_free_riders_between_hdnodes[timestep] += temp_fr[1] / average
                result_free_riders_between_vac[timestep] += temp_fr[2] / average
                result_free_riders_between_vac_rate[timestep] += temp_fr[3] / average
                for state_index in range(8):
                    result_states_trans[state_index][timestep] += temp_state[0][state_index] / average
                    result_top_nodes_state_trans[state_index][timestep] += temp_state[1][state_index] / average

                cc_result = Analyse.AnalyseConnectedComponent(total_attribute, G, seeds, infect_nums)
                result_cc_number[timestep] += cc_result[0] / average
                result_with_seed_cc_size[timestep] += cc_result[1] / average
                result_with_seed_cc_infect_rate[timestep] += cc_result[2] / average
                result_with_seed_cc_allcc_size_rate[timestep] += cc_result[3] / average
                result_max_cc_size[timestep] += cc_result[4] / average

                Analyse.GetDynamicGraphData(total_attribute, seeds, timestep, size, interval, score)

            mySIR.UpdateStrategy(total_attribute, size)

        if analyse:
            for i in range(len(interval)):
                if len(interval[i]) != 0:
                    interval[i] += '>'
            for i in range(len(score)):
                if len(score[i]) != 0:
                    score[i] += '>'
            df1 = pd.DataFrame({"id": id, "label": label, "interval": interval, "score": score})
            df2 = pd.DataFrame(edges, columns=['source', 'target'])
            
            # with pd.ExcelWriter(path+"_dynamic network_{}.xlsx".format(aver_time)) as writer: # excel会存在单元格内容溢出的问题
            #     df1.to_excel(excel_writer=writer, sheet_name="node", index=None)
            #     df2.to_excel(excel_writer=writer, sheet_name="egde", index=None) 
            
            df1.to_csv(path+"_dynamic network_node_{}.csv".format(aver_time), index=False)
            df2.to_csv(path+"_dynamic network_edge_{}.csv".format(aver_time), index=False)

    # 统计social cost
    for timestep in range(iteration):
        result_sc[timestep] += (result_vac[timestep] * c + result_infect[timestep]) * size
        result_total_sc[timestep] += sum(result_sc[:timestep+1])
        result_average_sc[timestep] += result_total_sc[timestep] / (timestep + 1)

    data = [result_vac, result_infect, result_S_infect, result_sc, result_average_sc, result_total_sc]
    Plot.Visualization(list(range(iteration)), data[:3], 'time', 'proportion', 'vac-infect',
                       ['vac', 'infect', 'S_infect'], path+'_vac-infect.jpg', multiple=True)
    Plot.Visualization(list(range(iteration)), data[3:5], 'time', 'social cost', 'sc-average sc',
                       ['sc', 'average_sc'], path+'_sc.jpg', multiple=True)
    Plot.Visualization(list(range(iteration)), data[-1], 'time', 'social cost', 'total social cost',
                       None, path+'_total sc.jpg', multiple=False)
    SaveData(data, ['vac', 'infect', 'S_infect', 'sc', 'average sc', 'total sc'], list(range(1, iteration+1)), r'name\t', path+'.csv')

    if analyse:
        data = [result_hdnodes_vac, result_seeds_distribution, result_free_riders, result_free_riders_between_hdnodes,
                result_cc_number, result_with_seed_cc_size, result_with_seed_cc_infect_rate, result_with_seed_cc_allcc_size_rate,
                result_max_cc_size, result_free_riders_between_vac, result_free_riders_between_vac_rate]
        many_row_name = ['top nodes vac', 'seeds distribution', 'free riders', 'free riders between hdnodes',
                        'cc number', 'with seed cc size', 'with seed cc infect rate', 'with seed cc/all cc',
                        'max_cc', 'the rate of free riders between vac nodes', 'the number of free riders rate between vac nodes']
        SaveData(data, many_row_name, list(range(1, iteration+1)), r'name\t', path + '_analysis_data.csv')
        row_name = ['s(nv)_nv', 'r_nv', 's(nv)_v', 'r_v', 'v_v', 'nv_nv', 'v_nv', 'nv_v']
        SaveData(result_states_trans, row_name, list(range(1, iteration+1)), r'name\t', path + '_analysis_statetrans_data.csv')
        SaveData(result_top_nodes_state_trans, row_name, list(range(1, iteration + 1)), r'name\t', path + '_analysis_top_nodes_statetrans_data.csv')

        Plot.Visualization(list(range(iteration)), [result_vac, result_hdnodes_vac], 'time', 'proportion', 'vac',
                           ['vac', 'hd-vac'], path + '_vac.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), result_seeds_distribution, 'time', 'proportion', 'seeds distribution',
                           None, path + '_seeds distribution.jpg', multiple=False)
        Plot.Visualization(list(range(iteration)), [result_free_riders, result_free_riders_between_hdnodes, result_free_riders_between_vac],
                           'time', 'number', 'free rider number', ['n_fr', 'n_hdn_fr', 'n_vac_fr'], path + '_free rider number.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), result_free_riders_between_vac_rate, 'time', 'proportion', 'free rider rate',
                           None, path + '_free rider rate.jpg', multiple=False)
        Plot.Visualization(list(range(iteration)), result_cc_number, 'time', 'number', 'cc number',
                           None, path + '_cc number.jpg', multiple=False)
        Plot.Visualization(list(range(iteration)), result_with_seed_cc_size, 'time', 'number', 'cc with seed size',
                           None, path + '_cc with seed size.jpg', multiple=False)
        Plot.Visualization(list(range(iteration)), result_with_seed_cc_allcc_size_rate, 'time', 'proportion', 'with seed cc/all cc',
                           None, path + '_with seed cc__all cc.jpg', multiple=False)
        Plot.Visualization(list(range(iteration)), result_with_seed_cc_infect_rate, 'time', 'proportion', 'with_seed_cc_infect_rate',
                           None, path + '_with seed cc infect rate.jpg', multiple=False)
        Plot.Visualization(list(range(iteration)), result_max_cc_size, 'time', 'number', 'max cc size',
                           None, path + '_max cc size.jpg', multiple=False)
        Plot.Visualization(list(range(iteration)), result_states_trans, 'time', 'proportion', 'statetrans',
                           row_name, path + '_statetrans.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), result_top_nodes_state_trans, 'time', 'proportion', 'top nodes statetrans',
                           row_name, path + '_top nodes statetrans.jpg', multiple=True)

    return data


def run_x_parameter(args, x_p_name, list_p):
    # parameters
    size = args.size
    average_degree = args.average_degree
    net_type = args.net_type
    r = args.r
    g = args.g
    c = args.c
    infectseeds = args.infectseeds
    k = args.k
    attack = args.attack

    total_attribute = np.asarray([[0] * size for i in range(5)], dtype='float32')

    # results
    average = args.average
    iteration = args.iteration
    accumulation = 100
    path = r"result\\" + net_type + " " + attack + "_x_{}_k{}".format(x_p_name, k)
    xp_vac = []
    xp_infect = []
    xp_sc = []
    xp_average_sc = []
    xp_total_sc = []


    for p_i in list_p:
        c = p_i # !!!
        result_vac = [0 for i in range(iteration)]
        result_infect = [0 for i in range(iteration)]
        result_sc = [0 for i in range(iteration)]
        result_average_sc = [0 for i in range(iteration)]
        result_total_sc = [0 for i in range(iteration)]

        for aver_time in trange(average):
            # Initalize Graph
            G = mySIR.Graph_generation(size, average_degree, net_type)
            neighborsList, neighborsArray = mySIR.GraphToArray(G)
            if attack == 'CC' or attack == 'BC':
                total_attribute[1][:] = 1
                nonvac_seeds = mySIR.ChooseInfectSeeds(total_attribute, G, size, infectseeds, attack)
            mySIR.InitalizationVac(total_attribute, size)
            for timestep in range(iteration):
                if attack == 'CC' or attack == 'BC':
                    if sum(total_attribute[1]) == size:  # non-vac
                        seeds = nonvac_seeds
                    else:
                        seeds = mySIR.ChooseInfectSeeds(total_attribute, G, size, infectseeds, attack)
                else:
                    seeds = mySIR.ChooseInfectSeeds(total_attribute, G, size, infectseeds, attack)

                # SIR
                mySIR.InitalizationInfect(total_attribute, infectseeds, seeds, size)
                lamda = mySIR.TransitionRateSum(total_attribute, neighborsArray, size, r, g)
                infect_nums = mySIR.GillespieAlgorithm(total_attribute, neighborsArray, lamda, infectseeds, size, r, g)
                mySIR.CalCost(total_attribute, size, c)
                # Learning Dynamics
                result_vac[timestep] += (mySIR.NumOfVaccination(total_attribute) / size / average)
                result_infect[timestep] += (infect_nums / size / average)
                mySIR.Imitation(total_attribute, neighborsArray, size, k)
                mySIR.UpdateStrategy(total_attribute, size)

        # 统计social cost
        for timestep in range(iteration):
            result_sc[timestep] += (result_vac[timestep] * c + result_infect[timestep]) * size
            result_total_sc[timestep] += sum(result_sc[:timestep + 1])
            result_average_sc[timestep] += result_total_sc[timestep] / (timestep + 1)

        xp_vac.append(np.asarray(result_vac[-accumulation:]).sum() / accumulation)
        xp_infect.append(np.asarray(result_infect[-accumulation:]).sum() / accumulation)
        xp_sc.append(np.asarray(result_sc[-accumulation:]).sum() / accumulation)
        xp_average_sc.append(result_average_sc[-1])
        xp_total_sc.append(result_total_sc[-1])

    data = [xp_vac, xp_infect, xp_sc, xp_average_sc, xp_total_sc]
    Plot.Visualization(list_p, data[:2], 'c', 'proportion', 'vac-infect', ['vac', 'infect'], path+"_vac-infect.jpg", multiple=True)
    Plot.Visualization(list_p, data[2:4], 'time', 'social cost', 'sc-average sc', ['sc', 'average_sc'], path+'_sc.jpg', multiple=True)
    Plot.Visualization(list_p, data[-1], 'time', 'social cost', 'total social cost', None, path+'_total sc.jpg', multiple=False)
    SaveData(data, ['vac', 'infect', 'sc', 'average sc', 'total sc'], list_p, 'name\c', path+'.csv')

    return data


def run_compare_attact_x_time(args, compare_parameter):
    # parameters
    size = args.size
    average_degree = args.average_degree
    net_type = args.net_type
    r = args.r
    g = args.g
    c = args.c
    infectseeds = args.infectseeds # number
    k = args.k
    attack = args.attack
    analyse = args.analyse
    compare_attack = compare_parameter

    total_attribute = np.asarray([[0] * size for i in range(5)], dtype='float32')

    # results
    average = args.average
    iteration = args.iteration
    path = r"result\\" + net_type + "_" + attack + '-' + compare_attack + "_c{}".format(c) + "_time_k{}".format(k)
    result_vac = [0 for i in range(iteration)]
    result_infect = [0 for i in range(iteration)]
    result_S_infect = [0 for i in range(iteration)]
    result_sc = [0 for i in range(iteration)]
    result_average_sc = [0 for i in range(iteration)]
    result_total_sc = [0 for i in range(iteration)]

    if analyse:
        top_n_node = 20
        result_hdnodes_vac = [0 for i in range(iteration)]
        result_seeds_distribution = [0 for i in range(iteration)]
        result_cc_number = [0 for i in range(iteration)]
        result_with_seed_cc_size = [0 for i in range(iteration)]
        result_with_seed_cc_infect_rate = [0 for i in range(iteration)]
        result_with_seed_cc_allcc_size_rate = [0 for i in range(iteration)]
        result_max_cc_size = [0 for i in range(iteration)]
        result_free_riders = [0 for i in range(iteration)]
        result_free_riders_between_hdnodes = [0 for i in range(iteration)]
        result_free_riders_between_vac = [0 for i in range(iteration)]
        result_free_riders_between_vac_rate = [0 for i in range(iteration)]
        result_states_trans = [[0 for i in range(iteration)] for j in range(8)]
        result_top_nodes_state_trans = [[0 for i in range(iteration)] for j in range(8)]

        result_compare_vac = [0 for i in range(iteration)]
        result_compare_infect = [0 for i in range(iteration)]
        result_compare_S_infect = [0 for i in range(iteration)]
        result_compare_seeds_distribution = [0 for i in range(iteration)]
        result_compare_with_seed_cc_size = [0 for i in range(iteration)]
        result_compare_with_seed_cc_infect_rate = [0 for i in range(iteration)]
        result_compare_with_seed_cc_allcc_size_rate = [0 for i in range(iteration)]
        result_compare_max_cc_size = [0 for i in range(iteration)]
        result_compare_free_riders = [0 for i in range(iteration)]
        result_compare_free_riders_between_hdnodes = [0 for i in range(iteration)]
        result_compare_free_riders_between_vac = [0 for i in range(iteration)]
        result_compare_free_riders_between_vac_rate = [0 for i in range(iteration)]
        result_compare_states_trans = [[0 for i in range(iteration)] for j in range(8)]
        result_compare_top_nodes_state_trans = [[0 for i in range(iteration)] for j in range(8)]


    for aver_time in range(average):
        # Initalize Graph
        G = mySIR.Graph_generation(size, average_degree, net_type)
        # G = mySIR.Graph_generation(size, average_degree, net_type, seed=42) # !!!
        neighborsList, neighborsArray = mySIR.GraphToArray(G)
        mySIR.InitalizationVac(total_attribute, size)

        if analyse:
            edges = [[source, target] for source, target in G.edges()]
            id = [i for i in G.nodes()]
            label = id
            interval = ["" for i in range(size)]
            score = ["" for i in range(size)]
            compare_interval = ["" for i in range(size)]
            compare_score = ["" for i in range(size)]

        for timestep in trange(iteration):
            temp_total_attribute = total_attribute.copy()
            seeds = mySIR.ChooseInfectSeeds(total_attribute, G, size, infectseeds, attack)
            compare_seeds = mySIR.ChooseInfectSeeds(total_attribute, G, size, infectseeds, compare_attack)

            if analyse:
                result_hdnodes_vac[timestep] += Analyse.AnalyseVacHDnode(total_attribute, G, top_n_node) / average
                result_seeds_distribution[timestep] += Analyse.AnalyseSeedsDistribution(total_attribute, G, top_n_node, seeds) / average
                result_compare_seeds_distribution[timestep] += Analyse.AnalyseSeedsDistribution(temp_total_attribute, G, top_n_node, compare_seeds) / average

            # SIR
            mySIR.InitalizationInfect(total_attribute, infectseeds, seeds, size)
            lamda = mySIR.TransitionRateSum(total_attribute, neighborsArray, size, r, g)
            infect_nums = mySIR.GillespieAlgorithm(total_attribute, neighborsArray, lamda, infectseeds, size, r, g)
            mySIR.CalCost(total_attribute, size, c)
            # Learning Dynamics
            result_vac[timestep] += (mySIR.NumOfVaccination(total_attribute) / size / average)
            result_infect[timestep] += (infect_nums / size / average)
            if infect_nums != 0:
                result_S_infect[timestep] += (infect_nums / (size - mySIR.NumOfVaccination(total_attribute))) / average
            mySIR.Imitation(total_attribute, neighborsArray, size, k)

            # compare SIR
            mySIR.InitalizationInfect(temp_total_attribute, infectseeds, compare_seeds, size)
            lamda = mySIR.TransitionRateSum(temp_total_attribute, neighborsArray, size, r, g)
            temp_infect_nums = mySIR.GillespieAlgorithm(temp_total_attribute, neighborsArray, lamda, infectseeds, size, r, g)
            mySIR.CalCost(temp_total_attribute, size, c)
            # Learning Dynamics
            mySIR.Imitation(temp_total_attribute, neighborsArray, size, k)
            if timestep == 0:
                result_compare_vac[timestep] += (mySIR.NumOfVaccination(temp_total_attribute) / size / average)
            if timestep != iteration - 1:
                result_compare_vac[timestep+1] += (size - sum(temp_total_attribute[0])) / size / average
            result_compare_infect[timestep] += (temp_infect_nums / size / average)
            if temp_infect_nums != 0:
                result_compare_S_infect[timestep] += (temp_infect_nums / (size - mySIR.NumOfVaccination(total_attribute))) / average

            if analyse:
                temp_fr = Analyse.AnalyseFR(total_attribute, neighborsArray, G, size, top_n_node)
                temp_state = Analyse.AnalyseStateTransition(total_attribute, G, size, top_n_node)
                result_free_riders[timestep] += temp_fr[0] / average
                result_free_riders_between_hdnodes[timestep] += temp_fr[1] / average
                result_free_riders_between_vac[timestep] += temp_fr[2] / average
                result_free_riders_between_vac_rate[timestep] += temp_fr[3] / average
                for state_index in range(8):
                    result_states_trans[state_index][timestep] += temp_state[0][state_index] / average
                    result_top_nodes_state_trans[state_index][timestep] += temp_state[1][state_index] / average

                cc_result = Analyse.AnalyseConnectedComponent(total_attribute, G, seeds, infect_nums)
                result_cc_number[timestep] += cc_result[0] / average
                result_with_seed_cc_size[timestep] += cc_result[1] / average
                result_with_seed_cc_infect_rate[timestep] += cc_result[2] / average
                result_with_seed_cc_allcc_size_rate[timestep] += cc_result[3] / average
                result_max_cc_size[timestep] += cc_result[4] / average
                origin_max_cc = cc_result[4] / average

                Analyse.GetDynamicGraphData(total_attribute, seeds, timestep, size, interval, score)

                temp_state = Analyse.AnalyseStateTransition(temp_total_attribute, G, size, top_n_node)
                temp_fr = Analyse.AnalyseFR(temp_total_attribute, neighborsArray, G, size, top_n_node)
                result_compare_free_riders[timestep] += temp_fr[0] / average
                result_compare_free_riders_between_hdnodes[timestep] += temp_fr[1] / average
                result_compare_free_riders_between_vac[timestep] += temp_fr[2] / average
                result_compare_free_riders_between_vac_rate[timestep] += temp_fr[3] / average
                for state_index in range(8):
                    result_compare_states_trans[state_index][timestep] += temp_state[0][state_index] / average
                    result_compare_top_nodes_state_trans[state_index][timestep] += temp_state[1][state_index] / average

                cc_result = Analyse.AnalyseConnectedComponent(temp_total_attribute, G, compare_seeds, temp_infect_nums)
                result_compare_with_seed_cc_size[timestep] += cc_result[1] / average
                result_compare_with_seed_cc_infect_rate[timestep] += cc_result[2] / average
                result_compare_with_seed_cc_allcc_size_rate[timestep] += cc_result[3] / average

                if timestep == 0:
                    result_compare_max_cc_size[timestep] += origin_max_cc
                if timestep != iteration - 1:
                    result_compare_max_cc_size[timestep + 1] += Analyse.AnalyseConnectedComponent(temp_total_attribute, G, compare_seeds, temp_infect_nums, compare = True) / average

                Analyse.GetDynamicGraphData(temp_total_attribute, compare_seeds, timestep, size, compare_interval, compare_score)

            mySIR.UpdateStrategy(total_attribute, size)

        if analyse:
            for i in range(len(interval)):
                if len(interval[i]) != 0:
                    interval[i] += '>'
            for i in range(len(score)):
                if len(score[i]) != 0:
                    score[i] += '>'
            df1 = pd.DataFrame({"id": id, "label": label, "interval": interval, "score": score})
            df2 = pd.DataFrame(edges, columns=['source', 'target'])
            with pd.ExcelWriter(path+"_dynamic network_{}_{}.xlsx".format(attack, aver_time)) as writer:
                df1.to_excel(excel_writer=writer, sheet_name="node", index=None)
                df2.to_excel(excel_writer=writer, sheet_name="egde", index=None)

            # compare result
            for i in range(len(compare_interval)):
                if len(compare_interval[i]) != 0:
                    compare_interval[i] += '>'
            for i in range(len(compare_score)):
                if len(compare_score[i]) != 0:
                    compare_score[i] += '>'
            df1 = pd.DataFrame({"id": id, "label": label, "interval": compare_interval, "score": compare_score})
            df2 = pd.DataFrame(edges, columns=['source', 'target'])
            with pd.ExcelWriter(path+"_dynamic network_{}_{}.xlsx".format(compare_attack, aver_time)) as writer:
                df1.to_excel(excel_writer=writer, sheet_name="node", index=None)
                df2.to_excel(excel_writer=writer, sheet_name="egde", index=None)

    data = [result_vac, result_infect, result_S_infect, result_compare_vac, result_compare_infect, result_compare_S_infect]
    Plot.Visualization(list(range(iteration)), [result_vac, result_infect, result_compare_vac, result_compare_infect], 'time', 'proportion', 'vac-infect',
                       ['{}_vac'.format(attack), '{}_infect'.format(attack), '{}_vac'.format(compare_attack), '{}_infect'.format(compare_attack)],
                       path+'_vac-infect.jpg', multiple=True)
    Plot.Visualization(list(range(iteration)), [result_S_infect, result_compare_S_infect], 'time', 'proportion', 'S_infect',
                       ['{}_S_infect'.format(attack), '{}_S_infect'.format(compare_attack)],
                       path+'_S_infect.jpg', multiple=True)
    SaveData(data, ['{}_vac'.format(attack), '{}_infect'.format(attack), '{}_S_infect'.format(attack), '{}_vac'.format(compare_attack),
                    '{}_infect'.format(compare_attack), '{}_S_infect'.format(compare_attack)],
             list(range(1, iteration+1)), r'name\t', path+'.csv')

    for timestep in range(iteration):
        result_sc[timestep] += (result_vac[timestep] * c + result_infect[timestep]) * size
        result_total_sc[timestep] += sum(result_sc[:timestep+1])
        result_average_sc[timestep] += result_total_sc[timestep] / (timestep + 1)

    data = [result_sc, result_average_sc, result_total_sc]
    Plot.Visualization(list(range(iteration)), data[:2], 'time', 'social cost', 'sc-average sc',
                       ['sc', 'average_sc'], path+'_sc.jpg', multiple=True)
    Plot.Visualization(list(range(iteration)), data[-1], 'time', 'social cost', 'total social cost',
                       None, path+'_total sc.jpg', multiple=False)
    SaveData(data, ['sc', 'average sc', 'total sc'], list(range(1, iteration+1)), r'name\t', path+'_social cost.csv')


    if analyse:
        data = [result_hdnodes_vac, result_seeds_distribution, result_free_riders, result_free_riders_between_hdnodes,
                result_cc_number, result_with_seed_cc_size, result_with_seed_cc_infect_rate, result_with_seed_cc_allcc_size_rate,
                result_max_cc_size, result_free_riders_between_vac, result_free_riders_between_vac_rate]
        many_row_name = ['top nodes vac', 'seeds distribution', 'free riders', 'free riders between hdnodes',
                        'cc number', 'with seed cc size', 'with seed cc infect rate', 'with seed cc/all cc',
                        'max_cc_size', 'the number of free riders rate between vac nodes', 'the rate of free riders between vac nodes']
        SaveData(data, many_row_name, list(range(1, iteration+1)), r'name\t', path + '_analysis_data.csv')
        row_name = ['s(nv)_nv', 'r_nv', 's(nv)_v', 'r_v', 'v_v', 'nv_nv', 'v_nv', 'nv_v']
        SaveData(result_states_trans, row_name, list(range(1, iteration+1)), r'name\t', path + '_analysis_statetrans_data.csv')
        SaveData(result_top_nodes_state_trans, row_name, list(range(1, iteration + 1)), r'name\t', path + '_analysis_top_nodes_statetrans_data.csv')

        compare_data = [result_compare_seeds_distribution, result_compare_free_riders, result_compare_free_riders_between_hdnodes,
                        result_compare_with_seed_cc_size, result_compare_with_seed_cc_infect_rate, result_compare_with_seed_cc_allcc_size_rate,
                        result_compare_max_cc_size, result_compare_free_riders_between_vac_rate, result_compare_free_riders_between_vac]
        many_row_name = ['seeds distribution', 'free riders', 'free riders between hdnodes',
                        'with seed cc size', 'with seed cc infect rate', 'with seed cc/all cc',
                        'max_cc_size', 'the rate of free riders between vac nodes', 'the number of free riders rate between vac nodes']
        SaveData(compare_data, many_row_name, list(range(1, iteration + 1)), r'name\t', path + '_compare_analysis_data.csv')
        row_name = ['s(nv)_nv', 'r_nv', 's(nv)_v', 'r_v', 'v_v', 'nv_nv', 'v_nv', 'nv_v']
        SaveData(result_compare_states_trans, row_name, list(range(1, iteration+1)), r'name\t', path + '_compare_analysis_statetrans_data.csv')
        SaveData(result_compare_top_nodes_state_trans, row_name, list(range(1, iteration + 1)), r'name\t', path + '_compare_analysis_top_nodes_statetrans_data.csv')

        Plot.Visualization(list(range(iteration)), [result_vac, result_hdnodes_vac, result_compare_vac], 'time', 'proportion', 'vac',
                           ['{}_vac'.format(attack), 'hd-vac', '{}_vac'.format(compare_attack)], path + '_vac.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), [result_seeds_distribution, result_compare_seeds_distribution], 'time', 'proportion', 'seeds distribution',
                           [attack, compare_attack], path + '_seeds distribution.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), [result_free_riders, result_free_riders_between_hdnodes, result_free_riders_between_vac,
                            result_compare_free_riders, result_compare_free_riders_between_hdnodes, result_compare_free_riders_between_vac],
                           'time', 'number', 'free rider number', ['n_fr', 'n_hdn_fr', 'n_vac_fr', 'n_c_fr', 'n_c_hdn_fr', 'n_c_vac_fr'],
                           path + '_free rider number.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), [result_free_riders_between_vac_rate, result_compare_free_riders_between_vac_rate], 'time', 'proportion', 'free rider rate',
                           [attack, compare_attack], path + '_free rider rate.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), result_cc_number, 'time', 'number', 'cc number',
                           None, path + '_cc number.jpg', multiple=False)
        Plot.Visualization(list(range(iteration)), [result_with_seed_cc_size, result_compare_with_seed_cc_size], 'time', 'number', 'cc with seed size',
                           [attack, compare_attack], path + '_cc with seed size.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), [result_with_seed_cc_allcc_size_rate, result_compare_with_seed_cc_allcc_size_rate], 'time', 'proportion', 'with seed cc/all cc',
                           [attack, compare_attack], path + '_with seed cc__all cc.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), [result_with_seed_cc_infect_rate, result_compare_with_seed_cc_infect_rate], 'time', 'proportion', 'with_seed_cc_infect_rate',
                           [attack, compare_attack], path + '_with seed cc infect rate.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), [result_max_cc_size, result_compare_max_cc_size], 'time', 'number', 'max cc size',
                           [attack, compare_attack], path + '_max cc size.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), result_states_trans, 'time', 'proportion', 'statetrans',
                           row_name, path + '_statetrans.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), result_top_nodes_state_trans, 'time', 'proportion', 'top nodes statetrans',
                           row_name, path + '_top nodes statetrans.jpg', multiple=True)
        row_name = ['v_v_{}'.format(attack), 'nv_nv{}'.format(attack), 'v_nv{}'.format(attack), 'nv_v{}'.format(attack), 'v_v_{}'.format(compare_attack), 'nv_nv{}'.format(compare_attack), 'v_nv{}'.format(compare_attack), 'nv_v{}'.format(compare_attack)]
        Plot.Visualization(list(range(iteration)), result_states_trans[4:]+result_compare_states_trans[4:], 'time', 'proportion', 'statetrans',
                           row_name, path + '_compare statetrans.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), result_top_nodes_state_trans[4:]+result_compare_top_nodes_state_trans[4:], 'time', 'proportion', 'top nodes statetrans',
                           row_name, path + '_compare top nodes statetrans.jpg', multiple=True)

    return data


def run_x_time_resilience(args, disrupt_start_time, disrupt_time, disrupt_attack):
    # parameters
    size = args.size
    average_degree = args.average_degree
    net_type = args.net_type
    r = args.r
    g = args.g
    c = args.c
    infectseeds = args.infectseeds  # number
    k = args.k
    attack = args.attack
    analyse = args.analyse
    compare_attack = disrupt_attack

    total_attribute = np.asarray([[0] * size for i in range(5)], dtype='float32')

    # results
    average = args.average
    iteration = args.iteration
    path = r"result\\" + net_type + "_resilience_" + attack + '{}--{}'.format(disrupt_start_time, disrupt_start_time + disrupt_time) + '_{}'.format(compare_attack) + "_c{}".format(c) + "_time_k{}".format(k)
    result_vac = [0 for i in range(iteration)]
    result_infect = [0 for i in range(iteration)]
    result_sc = [0 for i in range(iteration)]

    if analyse:
        top_n_node = 20
        result_cc_number = [0 for i in range(iteration)]
        result_with_seed_cc_size = [0 for i in range(iteration)]
        result_with_seed_cc_infect_rate = [0 for i in range(iteration)]
        result_with_seed_cc_allcc_size_rate = [0 for i in range(iteration)]
        result_free_riders = [0 for i in range(iteration)]
        result_free_riders_between_vac = [0 for i in range(iteration)]
        result_free_riders_between_vac_rate = [0 for i in range(iteration)]

        result_compare_vac = [0 for i in range(iteration)]
        result_compare_infect = [0 for i in range(iteration)]
        result_compare_sc = [0 for i in range(iteration)]
        result_compare_cc_number = [0 for i in range(iteration)]
        result_compare_with_seed_cc_size = [0 for i in range(iteration)]
        result_compare_with_seed_cc_infect_rate = [0 for i in range(iteration)]
        result_compare_with_seed_cc_allcc_size_rate = [0 for i in range(iteration)]
        result_compare_free_riders = [0 for i in range(iteration)]
        result_compare_free_riders_between_vac = [0 for i in range(iteration)]
        result_compare_free_riders_between_vac_rate = [0 for i in range(iteration)]


    for aver_time in range(average):
        # Initalize Graph
        G = mySIR.Graph_generation(size, average_degree, net_type)
        # G = mySIR.Graph_generation(size, average_degree, net_type, seed=42) # !!!
        neighborsList, neighborsArray = mySIR.GraphToArray(G)
        if compare_attack == 'CC' or compare_attack == 'BC':
            total_attribute[1][:] = 1
            nonvac_seeds = mySIR.ChooseInfectSeeds(total_attribute, G, size, infectseeds, attack)
        mySIR.InitalizationVac(total_attribute, size)

        for timestep in trange(iteration):
            if timestep == disrupt_start_time:
                temp_total_attribute = total_attribute.copy()

            if timestep < disrupt_start_time:
                # SIR
                seeds = mySIR.ChooseInfectSeeds(total_attribute, G, size, infectseeds, attack)
                mySIR.InitalizationInfect(total_attribute, infectseeds, seeds, size)
                lamda = mySIR.TransitionRateSum(total_attribute, neighborsArray, size, r, g)
                infect_nums = mySIR.GillespieAlgorithm(total_attribute, neighborsArray, lamda, infectseeds, size, r, g)
                mySIR.CalCost(total_attribute, size, c)
                result_vac[timestep] += (mySIR.NumOfVaccination(total_attribute) / size / average)
                result_compare_vac[timestep] = result_vac[timestep]
                result_infect[timestep] += (infect_nums / size / average)
                result_compare_infect[timestep] = result_infect[timestep]
                mySIR.Imitation(total_attribute, neighborsArray, size, k)
                if analyse:
                    # 分析搭便车，个体状态转移
                    temp_fr = Analyse.AnalyseFR(total_attribute, neighborsArray, G, size, top_n_node)
                    result_free_riders[timestep] += temp_fr[0] / average
                    result_compare_free_riders[timestep] = result_free_riders[timestep]
                    result_free_riders_between_vac[timestep] += temp_fr[2] / average
                    result_compare_free_riders_between_vac[timestep] = result_free_riders_between_vac[timestep]
                    result_free_riders_between_vac_rate[timestep] += temp_fr[3] / average
                    result_compare_free_riders_between_vac_rate[timestep] = result_free_riders_between_vac_rate[timestep]
                    # 传染源在连通分量的传染情况
                    cc_result = Analyse.AnalyseConnectedComponent(total_attribute, G, seeds, infect_nums)
                    result_cc_number[timestep] += cc_result[0] / average
                    result_compare_cc_number[timestep] = result_cc_number[timestep]
                    result_with_seed_cc_size[timestep] += cc_result[1] / average
                    result_compare_with_seed_cc_size[timestep] = result_with_seed_cc_size[timestep]
                    result_with_seed_cc_infect_rate[timestep] += cc_result[2] / average
                    result_compare_with_seed_cc_infect_rate[timestep] = result_with_seed_cc_infect_rate[timestep]
                    result_with_seed_cc_allcc_size_rate[timestep] += cc_result[3] / average
                    result_compare_with_seed_cc_allcc_size_rate[timestep] = result_with_seed_cc_allcc_size_rate[timestep]
                mySIR.UpdateStrategy(total_attribute, size)

            if timestep >= disrupt_start_time and timestep <= (disrupt_start_time + disrupt_time):
                seeds = mySIR.ChooseInfectSeeds(total_attribute, G, size, infectseeds, attack)
                if compare_attack == 'CC' or compare_attack == 'BC':
                    if sum(temp_total_attribute[1]) == size:  # non-vac
                        compare_seeds = nonvac_seeds
                    else:
                        compare_seeds = mySIR.ChooseInfectSeeds(temp_total_attribute, G, size, infectseeds, compare_attack)
                else:
                    compare_seeds = mySIR.ChooseInfectSeeds(temp_total_attribute, G, size, infectseeds, compare_attack)

                # SIR
                mySIR.InitalizationInfect(total_attribute, infectseeds, seeds, size)
                lamda = mySIR.TransitionRateSum(total_attribute, neighborsArray, size, r, g)
                infect_nums = mySIR.GillespieAlgorithm(total_attribute, neighborsArray, lamda, infectseeds, size, r, g)
                mySIR.CalCost(total_attribute, size, c)
                # Learning Dynamics
                result_vac[timestep] += (mySIR.NumOfVaccination(total_attribute) / size / average)
                result_infect[timestep] += (infect_nums / size / average)
                mySIR.Imitation(total_attribute, neighborsArray, size, k)

                # compare SIR
                mySIR.InitalizationInfect(temp_total_attribute, infectseeds, compare_seeds, size)
                lamda = mySIR.TransitionRateSum(temp_total_attribute, neighborsArray, size, r, g)
                temp_infect_nums = mySIR.GillespieAlgorithm(temp_total_attribute, neighborsArray, lamda, infectseeds,
                                                            size, r, g)
                mySIR.CalCost(temp_total_attribute, size, c)
                # Learning Dynamics
                result_compare_vac[timestep] += (mySIR.NumOfVaccination(temp_total_attribute) / size / average)
                result_compare_infect[timestep] += (temp_infect_nums / size / average)
                mySIR.Imitation(temp_total_attribute, neighborsArray, size, k)

                if analyse:
                    temp_fr = Analyse.AnalyseFR(total_attribute, neighborsArray, G, size, top_n_node)
                    result_free_riders[timestep] += temp_fr[0] / average
                    result_free_riders_between_vac[timestep] += temp_fr[2] / average
                    result_free_riders_between_vac_rate[timestep] += temp_fr[3] / average

                    cc_result = Analyse.AnalyseConnectedComponent(total_attribute, G, seeds, infect_nums)
                    result_cc_number[timestep] += cc_result[0] / average
                    result_with_seed_cc_size[timestep] += cc_result[1] / average
                    result_with_seed_cc_infect_rate[timestep] += cc_result[2] / average
                    result_with_seed_cc_allcc_size_rate[timestep] += cc_result[3] / average

                    # compare attack
                    temp_fr = Analyse.AnalyseFR(temp_total_attribute, neighborsArray, G, size, top_n_node)
                    result_compare_free_riders[timestep] += temp_fr[0] / average
                    result_compare_free_riders_between_vac[timestep] += temp_fr[2] / average
                    result_compare_free_riders_between_vac_rate[timestep] += temp_fr[3] / average

                    cc_result = Analyse.AnalyseConnectedComponent(temp_total_attribute, G, compare_seeds,
                                                                  temp_infect_nums)
                    result_compare_cc_number[timestep] += cc_result[0] / average
                    result_compare_with_seed_cc_size[timestep] += cc_result[1] / average
                    result_compare_with_seed_cc_infect_rate[timestep] += cc_result[2] / average
                    result_compare_with_seed_cc_allcc_size_rate[timestep] += cc_result[
                                                                                 3] / average

                mySIR.UpdateStrategy(total_attribute, size)
                mySIR.UpdateStrategy(temp_total_attribute, size)


            if timestep > (disrupt_start_time + disrupt_time):
                seeds = mySIR.ChooseInfectSeeds(total_attribute, G, size, infectseeds, attack)
                compare_seeds = mySIR.ChooseInfectSeeds(temp_total_attribute, G, size, infectseeds, attack)

                # SIR
                mySIR.InitalizationInfect(total_attribute, infectseeds, seeds, size)
                lamda = mySIR.TransitionRateSum(total_attribute, neighborsArray, size, r, g)
                infect_nums = mySIR.GillespieAlgorithm(total_attribute, neighborsArray, lamda, infectseeds, size, r, g)
                mySIR.CalCost(total_attribute, size, c)
                # Learning Dynamics
                result_vac[timestep] += (mySIR.NumOfVaccination(total_attribute) / size / average)
                result_infect[timestep] += (infect_nums / size / average)
                mySIR.Imitation(total_attribute, neighborsArray, size, k)

                # compare SIR
                mySIR.InitalizationInfect(temp_total_attribute, infectseeds, compare_seeds, size)
                lamda = mySIR.TransitionRateSum(temp_total_attribute, neighborsArray, size, r, g)
                temp_infect_nums = mySIR.GillespieAlgorithm(temp_total_attribute, neighborsArray, lamda, infectseeds,
                                                            size, r, g)
                mySIR.CalCost(temp_total_attribute, size, c)
                # Learning Dynamics
                result_compare_vac[timestep] += (mySIR.NumOfVaccination(temp_total_attribute) / size / average)
                result_compare_infect[timestep] += (temp_infect_nums / size / average)
                mySIR.Imitation(temp_total_attribute, neighborsArray, size, k)

                if analyse:
                    temp_fr = Analyse.AnalyseFR(total_attribute, neighborsArray, G, size, top_n_node)
                    result_free_riders[timestep] += temp_fr[0] / average
                    result_free_riders_between_vac[timestep] += temp_fr[2] / average
                    result_free_riders_between_vac_rate[timestep] += temp_fr[3] / average

                    cc_result = Analyse.AnalyseConnectedComponent(total_attribute, G, seeds, infect_nums)
                    result_cc_number[timestep] += cc_result[0] / average
                    result_with_seed_cc_size[timestep] += cc_result[1] / average
                    result_with_seed_cc_infect_rate[timestep] += cc_result[2] / average
                    result_with_seed_cc_allcc_size_rate[timestep] += cc_result[3] / average

                    # compare attack
                    temp_fr = Analyse.AnalyseFR(temp_total_attribute, neighborsArray, G, size, top_n_node)
                    result_compare_free_riders[timestep] += temp_fr[0] / average
                    result_compare_free_riders_between_vac[timestep] += temp_fr[2] / average
                    result_compare_free_riders_between_vac_rate[timestep] += temp_fr[3] / average

                    cc_result = Analyse.AnalyseConnectedComponent(temp_total_attribute, G, compare_seeds,
                                                                  temp_infect_nums)
                    result_compare_cc_number[timestep] += cc_result[0] / average
                    result_compare_with_seed_cc_size[timestep] += cc_result[1] / average
                    result_compare_with_seed_cc_infect_rate[timestep] += cc_result[2] / average
                    result_compare_with_seed_cc_allcc_size_rate[timestep] += cc_result[
                                                                                 3] / average

                mySIR.UpdateStrategy(total_attribute, size)
                mySIR.UpdateStrategy(temp_total_attribute, size)

    for timestep in range(iteration):
        result_sc[timestep] += (result_vac[timestep] * c + result_infect[timestep]) * size
        result_compare_sc[timestep] += (result_compare_vac[timestep] * c + result_compare_infect[timestep]) * size
    data = [result_vac, result_infect, result_sc, result_compare_vac, result_compare_infect, result_compare_sc]
    SaveData(data, ['{}_vac'.format(attack), '{}_infect'.format(attack), '{}_sc'.format(attack), '{}_vac'.format(compare_attack),
                    '{}_infect'.format(compare_attack), '{}_sc'.format(compare_attack)],
             list(range(1, iteration + 1)), r'name\t', path + '.csv')
    Plot.Visualization(list(range(iteration)), [result_vac, result_compare_vac], 'time',
                       'proportion', 'vac',
                       [attack, compare_attack], path + '_vac.jpg',
                       multiple=True)
    Plot.Visualization(list(range(iteration)), [result_infect, result_compare_infect], 'time',
                       'proportion', 'infect',
                       [attack, compare_attack], path + '_infect.jpg',
                       multiple=True)
    Plot.Visualization(list(range(iteration)), [result_sc, result_compare_sc], 'time',
                       'social cost', 'SC',
                       [attack, compare_attack], path + '_sc.jpg',
                       multiple=True)

    if analyse:
        data = [result_free_riders,
                result_cc_number, result_with_seed_cc_size, result_with_seed_cc_infect_rate,
                result_with_seed_cc_allcc_size_rate,
                result_free_riders_between_vac, result_free_riders_between_vac_rate]
        many_row_name = ['free riders',
                         'cc number', 'with seed cc size', 'with seed cc infect rate', 'with seed cc/all cc',
                         'the number of free riders rate between vac nodes',
                         'the rate of free riders between vac nodes']
        SaveData(data, many_row_name, list(range(1, iteration + 1)), r'name\t', path + '_analysis_data.csv')


        compare_data = [result_compare_free_riders,
                        result_cc_number, result_compare_with_seed_cc_size, result_compare_with_seed_cc_infect_rate,
                        result_compare_with_seed_cc_allcc_size_rate,
                        result_compare_free_riders_between_vac_rate, result_compare_free_riders_between_vac]
        many_row_name = ['free riders',
                         'cc number', 'with seed cc size', 'with seed cc infect rate', 'with seed cc/all cc',
                         'the rate of free riders between vac nodes',
                         'the number of free riders rate between vac nodes']
        SaveData(compare_data, many_row_name, list(range(1, iteration + 1)), r'name\t',
                 path + '_compare_analysis_data.csv')


        Plot.Visualization(list(range(iteration)),
                           [result_free_riders, result_free_riders_between_vac,
                            result_compare_free_riders,
                            result_compare_free_riders_between_vac],
                           'time', 'number', 'free rider number',
                           ['n_fr', 'n_vac_fr', 'n_c_fr', 'n_c_vac_fr'],
                           path + '_free rider number.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)),
                           [result_free_riders_between_vac_rate, result_compare_free_riders_between_vac_rate],
                           'time', 'proportion', 'free rider rate',
                           [attack, compare_attack], path + '_free rider rate.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), [result_cc_number, result_compare_cc_number], 'time', 'number', 'cc number',
                           [attack, compare_attack], path + '_cc number.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)), [result_with_seed_cc_size, result_compare_with_seed_cc_size],
                           'time', 'number', 'cc with seed size',
                           [attack, compare_attack], path + '_cc with seed size.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)),
                           [result_with_seed_cc_allcc_size_rate, result_compare_with_seed_cc_allcc_size_rate],
                           'time', 'proportion', 'with seed cc/all cc',
                           [attack, compare_attack], path + '_with seed cc__all cc.jpg', multiple=True)
        Plot.Visualization(list(range(iteration)),
                           [result_with_seed_cc_infect_rate, result_compare_with_seed_cc_infect_rate], 'time',
                           'proportion', 'with_seed_cc_infect_rate',
                           [attack, compare_attack], path + '_with seed cc infect rate.jpg', multiple=True)
    return data


def run_upper_bound_x_parameter(args, x_p_name, list_p):
    # parameters
    size = args.size
    average_degree = args.average_degree
    net_type = args.net_type
    k = args.k

    total_attribute = np.asarray([[0] * size for i in range(5)], dtype='float32')

    # results
    average = args.average
    iteration = args.iteration
    accumulation = 100
    path = r"result\\" + net_type + " " + "upper_x_{}_k{}".format(x_p_name, k)
    xp_vac = []
    xp_infect = []
    xp_sc = []
    xp_average_sc = []
    xp_total_sc = []


    for p_i in list_p:
        c = p_i # !!!
        result_vac = [0 for i in range(iteration)]
        result_infect = [0 for i in range(iteration)]
        result_sc = [0 for i in range(iteration)]
        result_average_sc = [0 for i in range(iteration)]
        result_total_sc = [0 for i in range(iteration)]

        for aver_time in trange(average):
            # Initalize Graph
            G = mySIR.Graph_generation(size, average_degree, net_type)
            neighborsList, neighborsArray = mySIR.GraphToArray(G)
            mySIR.InitalizationVac(total_attribute, size)
            for timestep in range(iteration):
                infect_nums = mySIR.UpperBound(total_attribute, size)
                mySIR.CalCost(total_attribute, size, c)
                # Learning Dynamics
                result_vac[timestep] += (mySIR.NumOfVaccination(total_attribute) / size / average)
                result_infect[timestep] += (infect_nums / size / average)
                mySIR.Imitation(total_attribute, neighborsArray, size, k)
                mySIR.UpdateStrategy(total_attribute, size)

        for timestep in range(iteration):
            result_sc[timestep] += (result_vac[timestep] * c + result_infect[timestep]) * size
            result_total_sc[timestep] += sum(result_sc[:timestep + 1])
            result_average_sc[timestep] += result_total_sc[timestep] / (timestep + 1)

        xp_vac.append(np.asarray(result_vac[-accumulation:]).sum() / accumulation)
        xp_infect.append(np.asarray(result_infect[-accumulation:]).sum() / accumulation)
        xp_sc.append(np.asarray(result_sc[-accumulation:]).sum() / accumulation)
        xp_average_sc.append(result_average_sc[-1])
        xp_total_sc.append(result_total_sc[-1])

    data = [xp_vac, xp_infect, xp_sc, xp_average_sc, xp_total_sc]
    Plot.Visualization(list_p, data[:2], 'c', 'proportion', 'vac-infect', ['vac', 'infect'], path+"_vac-infect.jpg", multiple=True)
    Plot.Visualization(list_p, data[2:4], 'time', 'social cost', 'sc-average sc', ['sc', 'average_sc'], path+'_sc.jpg', multiple=True)
    Plot.Visualization(list_p, data[-1], 'time', 'social cost', 'total social cost', None, path+'_total sc.jpg', multiple=False)
    SaveData(data, ['vac', 'infect', 'sc', 'average sc', 'total sc'], list_p, 'name\c', path+'.csv')

    return data


def run_lower_bound_x_parameter(args, x_p_name, list_p):
    # parameters
    size = args.size
    average_degree = args.average_degree
    net_type = args.net_type
    k = args.k

    total_attribute = np.asarray([[0] * size for i in range(5)], dtype='float32')

    # results
    average = args.average
    iteration = args.iteration
    accumulation = 100
    path = r"result\\" + net_type + " " + "lower_x_{}_k{}".format(x_p_name, k)
    xp_vac = []
    xp_infect = []
    xp_sc = []
    xp_average_sc = []
    xp_total_sc = []

    for p_i in list_p:
        c = p_i  # !!!
        result_vac = [0 for i in range(iteration)]
        result_infect = [0 for i in range(iteration)]
        result_sc = [0 for i in range(iteration)]
        result_average_sc = [0 for i in range(iteration)]
        result_total_sc = [0 for i in range(iteration)]

        for aver_time in trange(average):
            # Initalize Graph
            G = mySIR.Graph_generation(size, average_degree, net_type)
            neighborsList, neighborsArray = mySIR.GraphToArray(G)
            mySIR.InitalizationVac(total_attribute, size)
            for timestep in range(iteration):
                infect_nums = mySIR.LowerBound(total_attribute, neighborsArray, size)
                mySIR.CalCost(total_attribute, size, c)
                # Learning Dynamics
                result_vac[timestep] += (mySIR.NumOfVaccination(total_attribute) / size / average)
                result_infect[timestep] += (infect_nums / size / average)
                mySIR.Imitation(total_attribute, neighborsArray, size, k)
                mySIR.UpdateStrategy(total_attribute, size)

        for timestep in range(iteration):
            result_sc[timestep] += (result_vac[timestep] * c + result_infect[timestep]) * size
            result_total_sc[timestep] += sum(result_sc[:timestep + 1])
            result_average_sc[timestep] += result_total_sc[timestep] / (timestep + 1)

        xp_vac.append(np.asarray(result_vac[-accumulation:]).sum() / accumulation)
        xp_infect.append(np.asarray(result_infect[-accumulation:]).sum() / accumulation)
        xp_sc.append(np.asarray(result_sc[-accumulation:]).sum() / accumulation)
        xp_average_sc.append(result_average_sc[-1])
        xp_total_sc.append(result_total_sc[-1])

    data = [xp_vac, xp_infect, xp_sc, xp_average_sc, xp_total_sc]
    Plot.Visualization(list_p, data[:2], 'c', 'proportion', 'vac-infect', ['vac', 'infect'], path + "_vac-infect.jpg",
                       multiple=True)
    Plot.Visualization(list_p, data[2:4], 'time', 'social cost', 'sc-average sc', ['sc', 'average_sc'],
                       path + '_sc.jpg', multiple=True)
    Plot.Visualization(list_p, data[-1], 'time', 'social cost', 'total social cost', None, path + '_total sc.jpg',
                       multiple=False)
    SaveData(data, ['vac', 'infect', 'sc', 'average sc', 'total sc'], list_p, 'name\c', path + '.csv')

    return data


def SaveData(data, rowname, colname, allname, filename):
    df = pd.DataFrame(data, columns=colname)
    df.insert(0, allname, rowname)
    df.to_csv(filename, index=False)


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=int(1000), help='Network size')
    parser.add_argument('--average_degree', type=int, default=int(4), help='Network average degree')
    # ba er ws la
    parser.add_argument('--net_type', type=str, default="ba", help='Network type')
    # r = 0.55 (BA);  r = 0.51 (ER);  r = 0.46 (LA)
    parser.add_argument('--r', type=float, default=float(0.55), help='The disease transmission rate')
    parser.add_argument('--g', type=float, default=float(1/3), help='The rate of recovery from infection')
    parser.add_argument('--c', type=float, default=float(0.35), help='Costs of vaccination')
    parser.add_argument('--infectseeds', type=int, default=int(10), help='Infected seeds')
    parser.add_argument('--k', type=float, default=float(1), help='The strength of selection') # 1 / 0.1
    parser.add_argument('--average', type=int, default=int(5), help='Average number of experiment')
    parser.add_argument('--iteration', type=int, default=int(3000), help='Number of experiment iterations')
    # Random; HDA; PG; KS; EC; CC; BC; None; induct; DR; CI
    parser.add_argument('--attack', type=str, default="CI", help='Propagation source selection method')
    parser.add_argument('--origin', type=int, default=False, help='Only use origin graph information')
    parser.add_argument('--analyse', type=int, default=True, help='Analyse module switch')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_common_args()
    print(args)
    # result = run_x_parameter(args, "c", np.linspace(0, 1, 21))
    result = run_x_time(args)
    # result = run_compare_attact_x_time(args, "Random")




def getEnvConfig(configIdx):
    if configIdx == 0:
        return {
            'EnvType': 'HYBRID',
            'N_user': 8,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'motion_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 100,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[0,0]],
            'user_map': [[0,1,2,3], [4,5,6,7]],
        }
    elif configIdx == 1:
        return {
            'EnvType': 'HYBRID',
            'N_user': 8,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'motion_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 100,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[1,1]],
            'user_map': [[0,1,2,3], [4,5,6,7]],
        }
    elif configIdx == 2:
        return {
            'EnvType': 'HYBRID',
            'N_user': 8,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'motion_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 100,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[2,2]],
            'user_map': [[0,1,2,3], [4,5,6,7]],
        }
    elif configIdx == 3:
        return {
            'EnvType': 'HYBRID',
            'N_user': 4,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'motion_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 50,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[0]],
            'user_map': [[0,1,2,3]],
        }
    elif configIdx == 4:
        return {
            'EnvType': 'HYBRID',
            'N_user': 4,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'motion_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 50,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[1]],
            'user_map': [[0,1,2,3]],
        }
    elif configIdx == 5:
        return {
            'EnvType': 'HYBRID',
            'N_user': 4,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'motion_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 50,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[2]],
            'user_map': [[0,1,2,3]],
        }
    elif configIdx == 6:
        return {
            'EnvType': 'HYBRID',
            'N_user': 20,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'motion_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 200,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[0,0,0,0,0]],
            'user_map': [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]],
        }
    elif configIdx == 7:
        return {
            'EnvType': 'HYBRID',
            'N_user': 20,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'motion_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 200,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[1,1,1,1,1]],
            'user_map': [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]],
        }
    elif configIdx == 8:
        return {
            'EnvType': 'HYBRID',
            'N_user': 20,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'motion_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 200,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[2,2,2,2,2]],
            'user_map': [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]],
        }
    elif configIdx == 9:
        return {
            'EnvType': 'HYBRID',
            'N_user': 32,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'motion_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 320,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[0,0,0,0,0,0,0,0]],
            'user_map': [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23], [24,25,26,27], [28,29,30,31]],
        }
    elif configIdx == 10:
        return {
            'EnvType': 'HYBRID',
            'N_user': 32,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'motion_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 320,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[1,1,1,1,1,1,1,1]],
            'user_map': [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23], [24,25,26,27], [28,29,30,31]],
        }
    elif configIdx == 11:
        return {
            'EnvType': 'HYBRID',
            'N_user': 32,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'motion_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 320,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[2,2,2,2,2,2,2,2]],
            'user_map': [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23], [24,25,26,27], [28,29,30,31]],
        }
    #========================================= Haptic Dataflow =========================================
    elif configIdx == 12:
        return {
            'EnvType': 'HYBRID',
            'N_user': 8,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'haptic_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 100,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[4,4]],
            'user_map': [[0,1,2,3], [4,5,6,7]],
        }
    elif configIdx == 13:
        return {
            'EnvType': 'HYBRID',
            'N_user': 8,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'haptic_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 100,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[5,5]],
            'user_map': [[0,1,2,3], [4,5,6,7]],
        }
    elif configIdx == 14:
        return {
            'EnvType': 'HYBRID',
            'N_user': 8,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'haptic_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 100,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[6,6]],
            'user_map': [[0,1,2,3], [4,5,6,7]],
        }
    elif configIdx == 15:
        return {
            'EnvType': 'HYBRID',
            'N_user': 20,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'haptic_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 200,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[4,4,4,4,4]],
            'user_map': [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]],
        }
    elif configIdx == 16:
        return {
            'EnvType': 'HYBRID',
            'N_user': 20,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'haptic_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 200,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[5,5,5,5,5]],
            'user_map': [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]],
        }
    elif configIdx == 17:
        return {
            'EnvType': 'HYBRID',
            'N_user': 20,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'haptic_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 200,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[6,6,6,6,6]],
            'user_map': [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]],
        }
    elif configIdx == 18:
        return {
            'EnvType': 'HYBRID',
            'N_user': 4,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'haptic_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 50,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[4]],
            'user_map': [[0,1,2,3]],
        }
    elif configIdx == 19:
        return {
            'EnvType': 'HYBRID',
            'N_user': 4,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'haptic_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 50,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[5]],
            'user_map': [[0,1,2,3]],
        }
    elif configIdx == 20:
        return {
            'EnvType': 'HYBRID',
            'N_user': 4,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'haptic_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 50,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[6]],
            'user_map': [[0,1,2,3]],
        }
    elif configIdx == 21:
        return {
            'EnvType': 'HYBRID',
            'N_user': 32,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'haptic_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 320,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[4,4,4,4,4,4,4,4]],
            'user_map': [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23], [24,25,26,27], [28,29,30,31]],
        }
    elif configIdx == 22:
        return {
            'EnvType': 'HYBRID',
            'N_user': 32,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'haptic_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 320,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[5,5,5,5,5,5,5,5]],
            'user_map': [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23], [24,25,26,27], [28,29,30,31]],
        }
    elif configIdx == 23:
        return {
            'EnvType': 'HYBRID',
            'N_user': 32,
            'LEN_window': 20,
            'N_aggregation': 4,
            'dataflow': 'haptic_1ms_20',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 320,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'sub_agents_idx': [[6,6,6,6,6,6,6,6]],
            'user_map': [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23], [24,25,26,27], [28,29,30,31]],
        }
    else:
        raise ValueError(f"Invalid configIdx: {configIdx}")


def visualizeEnvConfig(simParams):
    print(f"{'='*50}")
    print(f"Environment Configuration")
    print(f"{'='*50}")
    print(f"Environment Type:       {simParams['EnvType']}")
    print(f"Number of Users:        {simParams['N_user']}")
    print(f"Window Length:          {simParams['LEN_window']}")
    print(f"Dataflow:               {simParams['dataflow']}")
    print(f"Sigma List:             {simParams['sigma_list']}")
    print(f"Resource Bar:           {simParams['r_bar']}")
    print(f"Bandwidth:              {simParams['B']}")
    print(f"Sub Agents:             {simParams['sub_agents_idx']}")
    print(f"User Map:               {simParams['user_map']}")
    print(f"{'='*50}")
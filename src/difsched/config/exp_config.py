def getExpConfig(configIdx):
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
            'offline_dataset_idxs': [0,1,2],
        }
    elif configIdx == 1:
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
            'offline_dataset_idxs': [6,7,8],
        }
    elif configIdx == 2:
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
            'offline_dataset_idxs': [3,4,5],
        }
    elif configIdx == 3:
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
            'offline_dataset_idxs': [9,10,11],
        }
    elif configIdx == 4:
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
            'offline_dataset_idxs': [12,13,14],
        }
    elif configIdx == 5:
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
            'offline_dataset_idxs': [15,16,17],
        }
    elif configIdx == 6:
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
            'offline_dataset_idxs': [18,19,20],
        }
    elif configIdx == 7:
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
            'offline_dataset_idxs': [21,22,23],
        }
    else:
        raise ValueError(f"Invalid configIdx: {configIdx}")


def visualizeExpConfig(expParams): 
    print(f"EnvType: {expParams['EnvType']}")
    print(f"N_user: {expParams['N_user']}")
    print(f"LEN_window: {expParams['LEN_window']}")
    print(f"N_aggregation: {expParams['N_aggregation']}")
    print(f"dataflow: {expParams['dataflow']}")
    print(f"randomSeed: {expParams['randomSeed']}")
    print(f"r_bar: {expParams['r_bar']}")
    print(f"B: {expParams['B']}")
    print(f"sigma_list: {expParams['sigma_list']}")
    print(f"offline_dataset_idxs: {expParams['offline_dataset_idxs']}")
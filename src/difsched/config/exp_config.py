def getExpConfig(configIdx):
    if configIdx == 0:
        return {
            'EnvType': 'HYBRID',
            'N_user': 8,
            'LEN_window': 200,
            'N_aggregation': 4,
            'dataflow': 'thumb_fr',
            'randomSeed': 999,
            'r_bar': 4,
            'B': 40,
            'sigma_list': [0.7, 0.75, 0.8, 0.85, 0.9],
            'offline_dataset_idxs': [0,1,2],
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